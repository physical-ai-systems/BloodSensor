import torch

class BaF2Dispersion:
    """
    A class to calculate and plot the Refractive Index (RI) of Barium Fluoride (BaF2)
    using the 3-term Sellmeier dispersion equation.
    """
    def __init__(self):
     
        self.B1 = 0.643356
        self.C1 = 0.057789 ** 2
        self.B2 = 0.506762
        self.C2 = 0.10968 ** 2
        self.B3 = 3.8261
        self.C3 = 46.3864 ** 2

    def calculate_ri(self, wavelength_nm):
        """
        Computes the Refractive Index for a given wavelength or array of wavelengths.
        Input is in nanometers (nm), converted internally to micrometers (µm) for the formula.
        """
        wl_sq = wavelength_nm ** 2
        term1 = (self.B1 * wl_sq) / (wl_sq - self.C1)
        term2 = (self.B2 * wl_sq) / (wl_sq - self.C2)
        term3 = (self.B3 * wl_sq) / (wl_sq - self.C3)
        n_squared = 1.0 + term1 + term2 + term3
        return torch.sqrt(n_squared)



class SiDispersion:
    """
    A class to calculate and plot the Refractive Index (RI) of Silicon (Si)
    using the provided 3-term Sellmeier dispersion equation.
    """
    def __init__(self):
        self.B1 = 10.6684293
        self.C1 = 0.301516485 ** 2
        self.B2 = 0.0030434748
        self.C2 = 1.13475115 ** 2
        self.B3 = 1.54133408
        self.C3 = 1104.0 ** 2

    def calculate_ri(self, wavelength_nm):
        """
        Computes the Refractive Index for a given wavelength or array of wavelengths.
        Input is in nanometers (nm), converted internally to micrometers (µm).
        """
        wl_sq = wavelength_nm ** 2
        term1 = (self.B1 * wl_sq) / (wl_sq - self.C1)
        term2 = (self.B2 * wl_sq) / (wl_sq - self.C2)
        term3 = (self.B3 * wl_sq) / (wl_sq - self.C3)
        
        n_squared = 1.0 + term1 + term2 + term3
        return torch.sqrt(n_squared)