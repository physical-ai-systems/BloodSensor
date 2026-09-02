import torch
import copy
from PhotonicDataSetGeneration.MaterialsRI import BaF2Dispersion, SiDispersion
from PhotonicDataSetGeneration.WholeBloodRI import MalariaBloodRefractiveIndex

class PhotonicSensor:
    def __init__(self, N, angle, mode,device, materials, thicknesses, wavelength_nm,stage):
        self.N = N
        self.angle = angle
        self.mode = mode.upper()
        self.materials = materials
        self.device = device
        self.thicknesses = thicknesses
        self.wavelength_nm = wavelength_nm
        self.stage = stage
        self.k0 = 2 * torch.pi / (wavelength_nm).to(device)  

        self.ri_layer_a = BaF2Dispersion().calculate_ri(wavelength_nm)
        self.ri_layer_b = SiDispersion().calculate_ri(wavelength_nm)
        self.ri_malaria = MalariaBloodRefractiveIndex(wavelength_nm, stage=stage).get_effective_ri()
     

        self.layers = self.build_structure()


    def build_structure(self):
        layers = []
        for i in range(self.N):
            layers.append({ "material": self.materials["layer_a"]["name"], "thickness": self.thicknesses["layer_a"]})
            layers.append({ "material": self.materials["layer_b"]["name"], "thickness": self.thicknesses["layer_b"]})

        layers.append({ "material": self.stage, "thickness": self.thicknesses["defect"]})
        
        for i in range(self.N):
            layers.append({ "material": self.materials["layer_a"]["name"], "thickness": self.thicknesses["layer_a"]})
            layers.append({ "material": self.materials["layer_b"]["name"], "thickness": self.thicknesses["layer_b"]})

        return layers

    def get_coordinates(self):
        coordinate = 0
        for i, layer in enumerate(self.layers):
            if layer["thickness"] is not None:
                if isinstance(coordinate, int) and coordinate == 0 and isinstance(layer["thickness"], torch.Tensor):
                    coordinate = torch.tensor(0.0, device=layer["thickness"].device, dtype=layer["thickness"].dtype)

                layer["coordinates"] = copy.deepcopy([coordinate, coordinate + layer["thickness"]]) 
                coordinate = coordinate + layer["thickness"]
            else:
                raise ValueError("Layer thickness is not defined")

        return self.layers
    def perturbate_structure_thickness(self, delta_thickness_percentage):
        for layer in self.layers:
            if layer["thickness"] is not None:
                thickness_tensor = torch.as_tensor(layer["thickness"], device=self.device, dtype=torch.float32)
                delta_thickness = thickness_tensor * delta_thickness_percentage / 100.0
                layer["thickness"] = thickness_tensor + delta_thickness * (2.0 * torch.rand_like(thickness_tensor) - 1.0)
        self.get_coordinates()
        return self.layers

    def p_value(self, material, mode, theta):
        """Calculates optical admittance (momentum) term."""
        theta = torch.as_tensor(theta, dtype=torch.complex128, device=material.refractive_index.device)
        if mode == "TE":
            p = torch.cos(theta) * material.refractive_index
        elif mode == "TM":
            p = material.refractive_index / torch.cos(theta)
        else:
            raise ValueError("mode must be TE or TM")
        return p

    def snells_law(self, n1, n2, theta1):
        """Applies Snell's law: n1 * sin(theta1) = n2 * sin(theta2)."""
        theta1 = torch.as_tensor(theta1, dtype=torch.complex128, device=n1.device)
        sin_theta2 = (n1 / n2) * torch.sin(theta1)
        return torch.asin(sin_theta2)

    def transfer_matrix(self, layer, theta, mode):
        """Constructs the 2x2 characteristic matrix for a single optical layer."""
        device = layer.material.refractive_index.device
        theta = torch.as_tensor(theta, device=device)
        p = self.p_value(layer.material, mode, theta)
        
        M = torch.zeros((*self.k0.shape, 2, 2), device=device)
        
        # Delta phase accumulation
        delta = layer.thickness.to(device) * layer.material.refractive_index * torch.cos(theta)
        phase = self.k0 * delta
        
        M[..., 0, 0] = torch.cos(phase)
        M[..., 0, 1] = -(1j / p) * torch.sin(phase)
        M[..., 1, 0] = (-1j * p) * torch.sin(phase)
        M[..., 1, 1] = torch.cos(phase)
      
        return M

    def Reflectance(self, boundary_layers, transfer_matrix, theta, mode): 
        """Calculates power reflectance (R) and transmittance (T) from the system matrix."""
        theta_0 = torch.as_tensor(theta[0], device=self.k0.device)
        theta_f = torch.as_tensor(theta[-1], device=self.k0.device)

        jo = self.p_value(boundary_layers[0].material, mode, theta_0)
        js = self.p_value(boundary_layers[1].material, mode, theta_f)

        M = transfer_matrix

        denominator = (M[..., 0, 0] + M[..., 0, 1] * js) * jo + (M[..., 1, 0] + M[..., 1, 1] * js)
        numerator_r = (M[..., 0, 0] + M[..., 0, 1] * js) * jo - (M[..., 1, 0] + M[..., 1, 1] * js)
        
        r = numerator_r / denominator
        R = torch.abs(r) ** 2

        t_numerator = 2 * jo
        t = t_numerator / denominator
        T = (js.real / jo.real) * torch.abs(t) ** 2

        return R, T
    
    def Reflectance_from_layers(self, full_layer_stack, theta_0, mode):
        """Iterates through a stack of layers to compute the global matrix and R, T."""
        device = full_layer_stack[0].material.refractive_index.device
        theta = torch.as_tensor(theta_0, device=device)
        M = None

        # Loop purely physical layers (skip incident [0] and substrate [-1])
        for i in range(1, len(full_layer_stack) - 1):
            n_prev = full_layer_stack[i-1].material.refractive_index
            n_curr = full_layer_stack[i].material.refractive_index

            theta = self.snells_law(n_prev, n_curr, theta)

            M_layer = self.transfer_matrix(full_layer_stack[i], theta, mode)
            if M is None:
                M = M_layer
            else:
                M = torch.matmul(M, M_layer)

        # Handle final exit angle into substrate
        n_last = full_layer_stack[-2].material.refractive_index
        n_sub  = full_layer_stack[-1].material.refractive_index
        theta_f = self.snells_law(n_last, n_sub, theta)  

        R, T = self.Reflectance([full_layer_stack[0], full_layer_stack[-1]], M, [theta_0, theta_f], mode)
        return R, T
    
    def TransferMatrixMethod(self, stage):
        """Prepares the boundaries, updates the defect RI, and triggers the TMM."""
        device = self.wavelength_nm.device
        
        # Build layer stack with proper material objects
        full_stack = []
        
        # Incident layer
        inc_material = type('Material', (object,), {
            'refractive_index': torch.tensor(self.materials["surrounding"]["refractive_index"], device=device)
        })()
        full_stack.append(type('Layer', (object,), {'material': inc_material, 'thickness': None})())
        
        # Physical layers
        for layer in self.layers:
            if layer["material"] == self.stage:
                # Defect layer - use malaria RI
                material = type('Material', (object,), {'refractive_index': self.ri_malaria})()
            elif layer["material"] == "Si":
                material = type('Material', (object,), {'refractive_index': self.ri_layer_b})()
            elif layer["material"] == "BaF2":
                material = type('Material', (object,), {'refractive_index': self.ri_layer_a})()
            
            layer_obj = type('Layer', (object,), {
                'material': material,
                'thickness': layer["thickness"]
            })()
            full_stack.append(layer_obj)
        
        # Substrate layer
        sub_material = type('Material', (object,), {
            'refractive_index': torch.tensor(self.materials["substrate"]["refractive_index"], device=device)
        })()
        full_stack.append(type('Layer', (object,), {'material': sub_material, 'thickness': None})())
        
        # Define incident angle and execute solver
        theta_0 = torch.tensor(self.angle * torch.pi / 180.0, device=device)
        
        return self.Reflectance_from_layers(full_stack, theta_0, self.mode)