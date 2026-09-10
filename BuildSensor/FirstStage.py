import torch


class FirstStage():
    def __init__(self, layar_a, layar_b, layar_g, layar_d, d1, d2, dg, dd, n1, n2, wavelen, suround_in, suround_out, N, nAir, nSiO2, device):
        self.layar_a = layar_a
        self.layar_b = layar_b
        self.layar_g = layar_g
        self.layar_d = layar_d
        self.d1 = d1
        self.d2 = d2
        self.dg = dg
        self.dd = dd
        self.n1 = n1
        self.n2 = n2
        self.nAir = nAir
        self.nSiO2 = nSiO2
        self.wavelen = wavelen
        self.suround_in = suround_in
        self.suround_out = suround_out
        self.N = N
        self.device = device

    
    def ng(self, ngtype):
        wavelen = self.wavelen / 1000.0
        if ngtype == "na":
            return -0.51929*(wavelen**4) + 4.3531*(wavelen**3) - 12.984*(wavelen**2) + 15.789*(wavelen) - 2.0315 
        elif ngtype == "nc":
            return 1.2751*(wavelen**3) - 8.1973*(wavelen**2) + 16.168*wavelen - 2.9464 
    def nd(self, ndtype):
        if ndtype == "water":
            return 5.0063e-7 * self.wavelen + 2.2388
        elif ndtype == "healthy":
            return 1.6224e-7 * self.wavelen + 2.0199
        elif ndtype == "Thyroid":
            return 4.5184e-7 * self.wavelen + 2.1910



    def Structure(self, ngtype, ndtype): 
        layers = []
        for _ in range(self.N):
            layers.append((self.layar_a, self.d1, self.n1))
            layers.append((self.layar_b, self.d2, self.n2))

        layers.append((self.layar_g, self.dg,self.ng(ngtype)))
        layers.append((self.layar_d, self.dd,self.nd(ndtype)))
        layers.append((self.layar_g, self.dg, self.ng(ngtype)))

        for _ in range(self.N):
            layers.append((self.layar_a, self.d1, self.n1))
            layers.append((self.layar_b, self.d2, self.n2))

        return layers

    def get_layers(self, ngtype, ndtype):
        return self.Structure(ngtype, ndtype)

    def layer_matrix(self, n, d, wavelen):
        """
        Calculate the layer matrix for a given layer.

        Parameters:
        n (complex): Refractive index of the layer.
        d (float): Thickness of the layer.
        wavelen (float): Wavelength of the light.

        Returns:
        torch.Tensor: 2x2 layer matrix.
        """

        delta = 2 * torch.pi * n * d / wavelen
        M11 = torch.cos(delta)
        M12 = -1j * torch.sin(delta) / n
        M21 = -1j * n * torch.sin(delta)
        M22 = torch.cos(delta)
        return torch.stack(
    [
        torch.stack([M11, M12]),
        torch.stack([M21, M22]),
    ]
)

    def period_matrix(self):
        M1 = self.layer_matrix(self.n1, self.d1, self.wavelen)
        M2 = self.layer_matrix(self.n2, self.d2, self.wavelen)
        return torch.matmul(M1, M2)
    
    def total_matrix(self, ngtype, ndtype):
        AB_total = torch.eye(2, dtype=torch.complex128)
        M_total = torch.eye(2, dtype=torch.complex128)
        for _ in range(self.N):
            AB_total = torch.matmul(AB_total, self.period_matrix())
        layers = self.get_layers(ngtype, ndtype)
        G = self.layer_matrix(layers[6][2], self.dg, self.wavelen)
        D = self.layer_matrix(layers[7][2], self.dd, self.wavelen)
        M_total = torch.matmul(torch.matmul(torch.matmul(AB_total, G), D), torch.matmul(G, AB_total))
        return M_total

    def transmittance(self, ngtype, ndtype):

        M_total = self.total_matrix(ngtype, ndtype)

        # Incident medium: Air
        p0 = torch.as_tensor(
            self.nAir,
            dtype=torch.complex128,
            device=self.device
        )

        # Exit medium: SiO2
        pf = torch.as_tensor(
            self.nAir,
            dtype=torch.complex128,
            device=self.device
        )

        M11 = M_total[0, 0]
        M12 = M_total[0, 1]
        M21 = M_total[1, 0]
        M22 = M_total[1, 1]

        denominator = (
            M11
            + M12 * pf
            + p0 * M21
            + p0 * pf * M22
        )

        t = 2.0 * p0 / denominator

        T = (
            torch.real(pf) / torch.real(p0)
        ) * torch.abs(t) ** 2

        return torch.real(T)

    def run(self, ngtype, ndtype):
        return self.transmittance(ngtype, ndtype)
