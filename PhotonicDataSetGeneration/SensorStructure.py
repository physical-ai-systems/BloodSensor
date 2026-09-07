import torch

from MaterialsRI import (SiDispersion,SiO2Dispersion,SilverDrude,)

from WholeBloodRI import MalariaBloodRefractiveIndex


class MalariaPhotonicSensor:
    """
    1D photonic-crystal malaria sensor from:

    (Si/SiO2)^3 / Ag / Defect / Ag / (Si/SiO2)^3

    Embedded in air.

    Based on:
    "An Improved Optical Biosensor design using
    defect/metal multilayer photonic crystal
    for Malaria Diagnosis", 2022.
    """

    def __init__(self,wavelengths_nm, stage, angle_deg, mode, si_periods, reference_wavelength_nm, metal_thickness_nm, defect_thickness_nm,  device):

        self.device = device or wavelengths_nm.device
        self.wavelengths_nm = wavelengths_nm.to(self.device)
        self.stage = stage.lower()
        self.angle_deg = angle_deg
        self.angle_rad = (torch.tensor(angle_deg, dtype=torch.float64, device=self.device) * torch.pi / 180.0)
        self.mode = mode.upper()
        if self.mode not in ["TE", "TM"]:
            raise ValueError("mode must be TE or TM")
        self.N = si_periods
        self.reference_wavelength_nm = reference_wavelength_nm
        self.metal_thickness_nm = metal_thickness_nm
        self.defect_thickness_nm = defect_thickness_nm

        self.si_model = SiDispersion()
        self.sio2_model = SiO2Dispersion()
        self.ag_model = SilverDrude()


        self.n_si = self.si_model.calculate_ri(self.wavelengths_nm).to(self.device)
        self.n_sio2 = self.sio2_model.calculate_ri( self.wavelengths_nm).to(self.device)
        self.n_ag = self.ag_model.calculate_ri(self.wavelengths_nm).to(self.device)
        self.n_defect = MalariaBloodRefractiveIndex( self.wavelengths_nm, self.stage).get_effective_ri().to(self.device)
        self.n_air = torch.ones_like(self.wavelengths_nm).to(self.device)
        reference_wavelength = (torch.tensor( reference_wavelength_nm, dtype=torch.float64, device=self.device))

        n_si_reference = self.si_model.calculate_ri(reference_wavelength)
        n_sio2_reference = self.sio2_model.calculate_ri(reference_wavelength)
        self.si_thickness_nm = (reference_wavelength/ (4.0 * n_si_reference))
        self.sio2_thickness_nm = (reference_wavelength / (4.0 * n_sio2_reference))


    def snells_law(self, n1, n2, theta1):

        sin_theta2 = ( n1 * torch.sin(theta1) / n2)
        return torch.asin(sin_theta2)


    def optical_admittance(self, n, theta,):

        if self.mode == "TE":
            return n * torch.cos(theta)
        else:
            return n / torch.cos(theta)



    def layer_matrix(self, n, thickness_nm, theta,):

        wavelength_nm = self.wavelengths_nm
        k = (2.0* torch.pi * n * torch.cos(theta) / wavelength_nm)
        delta = k * thickness_nm
        p = self.optical_admittance(n, theta)

        M11 = torch.cos(delta)
        M12 = (-1j * torch.sin(delta)/ p)
        M21 = (-1j * p * torch.sin(delta))
        M22 = torch.cos(delta)

        M = torch.zeros((wavelength_nm.shape[0], 2, 2), dtype=torch.complex128, device=self.device)

        M[:, 0, 0] = M11
        M[:, 0, 1] = M12
        M[:, 1, 0] = M21
        M[:, 1, 1] = M22

        return M

    

    @staticmethod
    def multiply(M1, M2):

        return torch.matmul(M1, M2)

 

    def build_transfer_matrix(self):

        wavelength = self.wavelengths_nm

        # Incident medium = air
        n_previous = self.n_air
        theta_previous = torch.full_like(wavelength, self.angle_rad, dtype=torch.complex128)
        M_total = torch.eye(2, dtype=torch.complex128, device=self.device).unsqueeze(0).repeat(wavelength.shape[0],1,1)


        for _ in range(self.N):

            # ---- Si ----

            theta_si = self.snells_law(n_previous.to(torch.complex128), self.n_si.to(torch.complex128), theta_previous)
            M_si = self.layer_matrix(self.n_si.to(torch.complex128), self.si_thickness_nm, theta_si)
            M_total = self.multiply(M_total, M_si)
            n_previous = self.n_si
            theta_previous = theta_si

            # ---- SiO2 ----

            theta_sio2 = self.snells_law(n_previous.to(torch.complex128), self.n_sio2.to(torch.complex128), theta_previous)
            M_sio2 = self.layer_matrix(self.n_sio2.to(torch.complex128), self.sio2_thickness_nm, theta_sio2)
            M_total = self.multiply(M_total, M_sio2)

            n_previous = self.n_sio2
            theta_previous = theta_sio2

        # -------------------------------------------------
        # Ag
        # -------------------------------------------------

        theta_ag = self.snells_law(n_previous.to(torch.complex128), self.n_ag, theta_previous)
        M_ag1 = self.layer_matrix(self.n_ag, self.metal_thickness_nm, theta_ag)
        M_total = self.multiply( M_total, M_ag1)
        n_previous = self.n_ag
        theta_previous = theta_ag

        # -------------------------------------------------
        # Defect / blood
        # -------------------------------------------------

        theta_defect = self.snells_law(n_previous.to(torch.complex128), self.n_defect.to(torch.complex128),theta_previous)
        M_defect = self.layer_matrix(self.n_defect.to(torch.complex128),self.defect_thickness_nm,theta_defect)
        M_total = self.multiply( M_total, M_defect)
        n_previous = self.n_defect
        theta_previous = theta_defect

        # -------------------------------------------------
        # Ag
        # -------------------------------------------------

        theta_ag = self.snells_law(n_previous.to(torch.complex128), self.n_ag, theta_previous)
        M_ag2 = self.layer_matrix( self.n_ag, self.metal_thickness_nm, theta_ag)
        M_total = self.multiply(M_total, M_ag2)
        n_previous = self.n_ag
        theta_previous = theta_ag

        # -------------------------------------------------
        # Right (Si/SiO2)^3
        # -------------------------------------------------

        for _ in range(self.N):

            # ---- Si ----

            theta_si = self.snells_law(n_previous.to(torch.complex128), self.n_si.to(torch.complex128), theta_previous)
            M_si = self.layer_matrix(self.n_si.to(torch.complex128), self.si_thickness_nm, theta_si)
            M_total = self.multiply( M_total, M_si)
            n_previous = self.n_si
            theta_previous = theta_si

            # ---- SiO2 ----
            theta_sio2 = self.snells_law(n_previous.to(torch.complex128), self.n_sio2.to(torch.complex128), theta_previous)
            M_sio2 = self.layer_matrix(self.n_sio2.to(torch.complex128), self.sio2_thickness_nm, theta_sio2)
            M_total = self.multiply( M_total, M_sio2)
            n_previous = self.n_sio2
            theta_previous = theta_sio2

        return M_total

    # =====================================================
    # R / T / A
    # =====================================================

    def calculate_spectrum(self):

        M = self.build_transfer_matrix()

        # Air on both sides
        theta_incident = torch.full_like(self.wavelengths_nm, self.angle_rad, dtype=torch.complex128)
        theta_substrate = theta_incident
        p_i = self.optical_admittance( self.n_air.to(torch.complex128), theta_incident)
        p_s = self.optical_admittance(self.n_air.to(torch.complex128), theta_substrate)

        M11 = M[:, 0, 0]
        M12 = M[:, 0, 1]
        M21 = M[:, 1, 0]
        M22 = M[:, 1, 1]

        denominator = (M11 + M12 * p_s) * p_i + (M21 + M22 * p_s)
        numerator_r = (M11 + M12 * p_s) * p_i - (M21 + M22 * p_s)
        r = numerator_r / denominator
        R = torch.abs(r) ** 2
        t = (2.0 * p_i/ denominator)
        T = (p_s.real/ p_i.real) * torch.abs(t) ** 2
        A = 1.0 - R - T

        # Numerical cleanup
        R = torch.real(R)
        T = torch.real(T)
        A = torch.real(A)

        return R, T, A

    # =====================================================
    # Convenience
    # =====================================================

    def simulate(self):

        R, T, A = self.calculate_spectrum()

        return {
            "wavelength": self.wavelengths_nm,
            "RI": self.n_defect,
            "R": R,
            "T": T,
            "A": A,
        }