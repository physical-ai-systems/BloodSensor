import torch


class SiDispersion:
    """
    Silicon refractive index used in the 2022 malaria
    1D photonic-crystal paper.

    Wavelength input:
        nm

    Internally:
        wavelength is converted to micrometers.
    """

    def calculate_ri(self, wavelength_nm):
        wavelength_um = wavelength_nm / 1000.0
        wavelength_sq = wavelength_um ** 2

        # n_Si^2 = 1 + 0.0938 * lambda^2 / (lambda^2 - 0.00866)
        n_squared = (1.0 + (0.0938 * wavelength_sq)/ (wavelength_sq - 0.00866))

        return torch.sqrt(n_squared)


class SiO2Dispersion:
    """
    SiO2 refractive index used in the paper.

    Wavelength input:
        nm

    Internally:
        wavelength is converted to micrometers.
    """

    def calculate_ri(self, wavelength_nm):
        wavelength_um = wavelength_nm / 1000.0
        wavelength_sq = wavelength_um ** 2

        n_squared = (1.3107237 + (0.7935797 * wavelength_sq) / (wavelength_sq - 0.0109597) + (0.9237144 * wavelength_sq) / (wavelength_sq - 100.0))

        return torch.sqrt(n_squared)


class SilverDrude:
    """
    Complex refractive index of silver using the Drude model
    described in the malaria photonic-crystal paper.

    Paper parameters:
        plasma frequency = 2.18 PHz
        damping frequency = 4.35 THz

    Angular frequencies are used internally.
    """

    def __init__(self,  plasma_frequency_hz=2.18e15, damping_frequency_hz=4.35e12):
        self.plasma_frequency_hz = plasma_frequency_hz
        self.damping_frequency_hz = damping_frequency_hz

    def calculate_ri(self, wavelength_nm):

        c = 299792458.0
        wavelength_m = wavelength_nm * 1e-9
        omega = 2.0 * torch.pi * c / wavelength_m
        omega_p = 2.0 * torch.pi * self.plasma_frequency_hz
        gamma = 2.0 * torch.pi * self.damping_frequency_hz
        # Drude dielectric function
        epsilon = 1.0 - (omega_p ** 2 / (omega ** 2 + 1j * gamma * omega))
        # Complex refractive index
        n = torch.sqrt(epsilon)
        return n