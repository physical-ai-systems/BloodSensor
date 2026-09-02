import torch 

class PlasmaDispersion:
    """
    A class to calculate and plot the Refractive Index (RI) of plasma 
    using the Cauchy dispersion equation.
    """
    def __init__(self, A=1.3353, B=4.4048e3, C=-9.1925e7):
        self.A = A
        self.B = B
        self.C = C

    def calculate_ri(self, wavelength):
        """
        Computes the Refractive Index for a given wavelength or array of wavelengths.
        Wavelength is assumed to be in nanometers (nm).
        """
        return self.A + (self.B / wavelength**2) + (self.C / wavelength**4)
    

class RBCDispersion:
    """
    A class to calculate and plot the real Refractive Index (RI) of 
    healthy oxygenated Red Blood Cells (RBCs).
    """
    def __init__(self, base_ri=1.39732, numerator_const=6.662, denominator_offset=129.2, stage= "normal"):
        self.base_ri = base_ri
        self.numerator_const = numerator_const
        self.denominator_offset = denominator_offset
        self.stage = stage

        offsets = {
        "normal": 0.000,
        "ring": -0.007,
        "trophozoite": -0.019,
        "schizont": -0.029
    }
        self.offset = offsets.get(stage.lower())

    def calculate_ri(self, wavelength):
        """
        Computes the Refractive Index for a given wavelength or array of wavelengths.
        Wavelength is assumed to be in nanometers (nm).
        """
        return (self.base_ri + (self.numerator_const / (wavelength - self.denominator_offset)))+ self.offset


class WBCDispersion:
    """
    A class to calculate and plot the Refractive Index (RI) of 
    White Blood Cells (WBCs) based on the intracellular dry-mass density model.
    """
    def __init__(self, base_ri=1.35604, numerator_const=6.662, denominator_offset=129.2):
        self.base_ri = base_ri
        self.numerator_const = numerator_const
        self.denominator_offset = denominator_offset

    def calculate_ri(self, wavelength):
        """
        Computes the Refractive Index for a given wavelength or array of wavelengths.
        Wavelength is assumed to be in nanometers (nm).
        """
        return self.base_ri + (self.numerator_const / (wavelength - self.denominator_offset))


class PlateletDispersion:
    """
    A class to calculate and plot the Refractive Index (RI) of 
    platelets based on the intracellular dry-mass density model.
    """
    def calculate_ri(self, wavelength):
        """
        Computes the Refractive Index for a given wavelength or array of wavelengths.
        Wavelength is assumed to be in nanometers (nm).
        """
        return torch.full_like(wavelength, 1.39)


class BloodRefractiveIndex:
    """
    Calculate the effective refractive index of whole blood
    using the Lorentz-Lorenz effective-medium model over wavelength arrays.
    """

    def __init__(self):
        self.material = {}

    @staticmethod
    def lorentz_lorenz(ri):
        """
        Calculate the Lorentz-Lorenz parameter.
        Works with both scalar values and numpy arrays.
        """
        return (ri**2 - 1.0) / (ri**2 + 2.0)

    def add_material(self, name, components):
        """
        Add a blood condition.
        """
        self.material[name] = components

    def calculate(self, name):
        """
        Calculate the effective refractive index.
        """
        components = self.material[name]
        
        L_effective = 0.0

        for component in components.values():
            percentage = component["percentage"]
            ri = component["ri"]
            fraction = percentage / 100.0

            L_i = self.lorentz_lorenz(ri)
            L_effective += fraction * L_i

        n_effective = torch.sqrt((1.0 + 2.0 * L_effective) / (1.0 - L_effective))
        
        return n_effective

    def calculate_all(self):
        """
        Calculate effective RI for all blood conditions.
        """
        return {
            name: self.calculate(name)
            for name in self.material
        }



class MalariaBloodRefractiveIndex:
    def __init__(self, wavelengths, stage):
        self.wavelengths = wavelengths
        self.plasma_ri = PlasmaDispersion().calculate_ri(wavelengths)
        self.rbc_ri = RBCDispersion(stage=stage).calculate_ri(wavelengths)
        self.wbc_ri = WBCDispersion().calculate_ri(wavelengths)
        self.platelets_ri = PlateletDispersion().calculate_ri(wavelengths)

        self.calculator = BloodRefractiveIndex()

        self.calculator.add_material(f"Whole Blood {stage}", {
            "Plasma": {"percentage": 55.0, "ri": self.plasma_ri},
            "RBCs": {"percentage": 44.0, "ri": self.rbc_ri},
            "WBCs": {"percentage": 0.5, "ri": self.wbc_ri},
            "Platelets": {"percentage": 0.5, "ri": self.platelets_ri}
        })

        self.effective_ri_whole_blood = self.calculator.calculate(f"Whole Blood {stage}")

    def get_effective_ri(self):
        return self.effective_ri_whole_blood
