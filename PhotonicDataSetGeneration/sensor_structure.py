import torch
from PhotonicDataSetGeneration.MaterialsRI import BaF2Dispersion, SiDispersion
from PhotonicDataSetGeneration.WholeBloodRI import MalariaBloodRefractiveIndex

class PhotonicSensor:
    def __init__(self, N, angle, mode, perturbation, materials, thicknesses, wavelength_nm):
        self.N = N
        self.angle = angle
        self.mode = mode
        self.perturbation = perturbation
        self.materials = materials
        self.thicknesses = thicknesses
        self.ri_layer_a = BaF2Dispersion().calculate_ri(wavelength_nm)
        self.ri_layer_b = SiDispersion().calculate_ri(wavelength_nm)
        self.ri_normal_malaria = MalariaBloodRefractiveIndex(wavelength_nm, stage="normal").get_effective_ri()
        self.ri_ring_malaria = MalariaBloodRefractiveIndex(wavelength_nm, stage="ring").get_effective_ri()
        self.ri_trophozoite_malaria = MalariaBloodRefractiveIndex(wavelength_nm, stage="trophozoite").get_effective_ri()
        self.ri_schizont_malaria = MalariaBloodRefractiveIndex(wavelength_nm, stage="Schizont").get_effective_ri()

    def build_structure(self):
        layers = []
        for i in range(self.N):
            layers.append({ "material": self.materials.layer_a["name"], "thickness": self.thicknesses.layer_a})
            layers.append({ "material": self.materials.layer_b["name"], "thickness": self.thicknesses.layer_b})

        layers.append({ "material": self.materials.defect["name"], "thickness": self.thicknesses.defect})
        
        for i in range(self.N):
            layers.append({ "material": self.materials.layer_a["name"], "thickness": self.thicknesses.layer_a})
            layers.append({ "material": self.materials.layer_b["name"], "thickness": self.thicknesses.layer_b})

        return layers

    def 
