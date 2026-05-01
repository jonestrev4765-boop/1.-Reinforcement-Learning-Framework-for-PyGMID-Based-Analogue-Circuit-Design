"""
gm/ID design functions for a Current-Reuse Complementary Input (CRCI) amplifier.

Design equations from:
C. O'Donnell and D. O'Hare, "CCIA gm/ID Design," 2024. [Online]. Available: https://github.com/dreoilin/CCIA-Design
Located in "Current Reuse Amplifier.ipynb"

Calculations here can be adjusted to train on seperate topologies.
"""


from pygmid import Lookup as lk
import numpy as np

class CurrentReuse:

    def __init__(self, NCH, PCH,
                 VDD=1.2, kB=1.38e-23, T=300.15,
                 LP=0.5, LN=0.7,
                 Cfb=1e-12,         
                 CL=1e-12):          
        self.__NCH  = NCH
        self.__PCH  = PCH
        self.VDD    = VDD
        self.kB     = kB
        self.T      = T
        self.LPs    = LP
        self.LNs    = LN
        self.Cfb    = Cfb
        self.CL     = CL

    @staticmethod
    def _scalar(x):
        """Collapse any ndarray of size 1 to a plain Python float."""
        x = np.asarray(x)
        if x.size == 1:
            return float(x.flat[0])
        return x

    def _ensure_1d(self, *arrays):
        return [np.atleast_1d(np.asarray(a, dtype=float)) for a in arrays]

    def compute_all(self, GM_IDN, GM_IDP, Ibias, Cin, Fc_max=None):
        """
        Runs all required pygmid lookups and returns calculated values
        """
        
        [GM_IDN_a] = self._ensure_1d(GM_IDN)
        [GM_IDP_a] = self._ensure_1d(GM_IDP)

        VDS = self.VDD / 4.0

        # 1. Device widths (µm)
        WP = Ibias / self.__PCH.look_up('ID_W',  GM_ID=GM_IDP_a, VDS=VDS, L=self.LPs)
        WN = Ibias / self.__NCH.look_up('ID_W',  GM_ID=GM_IDN_a, VDS=VDS, L=self.LNs)
        WP_um = float(np.atleast_1d(WP).flat[0])   # µm
        WN_um = float(np.atleast_1d(WN).flat[0])   # µm

        # Gate capacitances
        CGSP = float(np.atleast_1d(WP_um * self.__PCH.look_up('CGS_W', GM_ID=GM_IDP_a, VDS=VDS, L=self.LPs)).flat[0])
        CGSN = float(np.atleast_1d(WN_um * self.__NCH.look_up('CGS_W', GM_ID=GM_IDN_a, VDS=VDS, L=self.LNs)).flat[0])
        Cg = CGSP + CGSN

        alpha = (Cin + Cg + self.Cfb) / Cin

        # Open-loop gain
        gds_N = float(np.atleast_1d(self.__NCH.look_up('GDS_ID', GM_ID=GM_IDN_a, L=self.LNs, VDS=VDS)).flat[0])
        gds_P = float(np.atleast_1d(self.__PCH.look_up('GDS_ID', GM_ID=GM_IDP_a, L=self.LPs, VDS=VDS)).flat[0])
        Av0 = (float(GM_IDP) + float(GM_IDN)) / (gds_P + gds_N)

        gammaN = float(np.atleast_1d(self.__NCH.gamma(GM_ID=GM_IDN_a, L=self.LNs, VDS=VDS)).flat[0])
        gammaP = float(np.atleast_1d(self.__PCH.gamma(GM_ID=GM_IDP_a, L=self.LPs, VDS=VDS)).flat[0])

        # Input-referred Thermal Noise (IRN)
        noise_num = 4.0 * self.kB * self.T * (gammaP + gammaN)
        noise_den = Ibias * (float(GM_IDP) + float(GM_IDN))
        VNin      = float(np.sqrt(alpha**2 * noise_num / noise_den))

        # Flicker corner 
        STH_N  = 4.0 * self.kB * self.T * gammaN
        STH_P  = 4.0 * self.kB * self.T * gammaP
        fco_N  = float(np.atleast_1d(self.__NCH.fco(GM_ID=GM_IDN_a, L=self.LNs, VDS=VDS)).flat[0])
        fco_P  = float(np.atleast_1d(self.__PCH.fco(GM_ID=GM_IDP_a, L=self.LPs, VDS=VDS)).flat[0])
        Fc = (STH_P * fco_P + STH_N * fco_N) / (STH_P + STH_N)

        # Power
        Power = self.VDD * 2.0 * Ibias

        beta   = self.Cfb / (Cin + self.Cfb + Cg)

        return {
            'Cg'         : Cg,
            'alpha'      : alpha,
            'Av0'        : Av0,
            'VNin'       : VNin,
            'VNin_nV'    : VNin * 1e9,
            'Fc'         : Fc,
            'Power'      : Power,
            'Power_uW'   : Power * 1e6,
            'beta'       : beta,
            'WP_um'      : WP_um,  
            'WN_um'      : WN_um,   
        }
