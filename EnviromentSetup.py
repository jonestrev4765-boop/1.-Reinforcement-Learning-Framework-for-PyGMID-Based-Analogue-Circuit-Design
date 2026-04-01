"""
RL environment for gm/ID optimisation.

Action space (4-dim, continuous in [-1, 1])

  [delta GM_IDN,  delta GM_IDP,  delta Ibias,  delta LN_LP_ratio]

Each action scaled by its own step size:

Mode
----
  0 - Training   : random starts, mismatch + area penalties active
  1 - Deployment : random starts, penalties disabled
  2 - Eval       : random starts, mismatch + area penalties active
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VNinsEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        amp_functions,
        # Action step sizes 
        gm_step_size          = 0.5,            # S/A per step
        ibias_step_size       = 1e-6,           # A per step (1 mA)
        ln_lp_step_size       = 0.05,           # ratio per step
        # Parameter ranges 
        GM_IDN_RANGE          = (15.0, 24.0),   # S/A
        GM_IDP_RANGE          = (15.0, 24.0),   # S/A
        IBIAS_RANGE           = (5e-6, 50e-6),  # A
        LN_LP_RANGE           = (1.0,  2.0),    # ratio
        # Fixed circuit parameters 
        LP                    = 1.0,            # um  PMOS length, fixed
        Cin                   = 10e-12,         # F
        Fc_max                 = 50e3,          # Hz
        # Area constraint 
        area_budget_um2       = 500,            # um^2 soft budget (W*L, input pair)
        area_weight           = 0.8,            # penalty weight above budget
        # Noise targets 
        vnin_target           = 4.5,            # nV/rtHz terminal success
        vnin_reference        = 10.0,           # nV/rtHz reward normalisation
        vnin_gate_threshold   = 6.0,            # nV/rtHz gate on fc reward
        # Reward weights 
        vnin_weight           = 0.8,
        power_weight          = 0.3,
        freq_weight           = 0.6,
        terminal_bonus        = 1.0,
        # Power reference 
        ibias_ref             = 30e-6,          # A
        # Mismatch penalty 
        mismatch_threshold    = 4.0,            # Prevents gmid values from being too far apart
        # fc reward scale 
        fc_scale              = 150e3,          # Hz
        # Mode 
        mode                  = 0,
        render_mode           = None,
    ):
        super().__init__()

        self.amp_functions   = amp_functions
        self.gm_step_size    = gm_step_size
        self.ibias_step_size = ibias_step_size
        self.ln_lp_step_size = ln_lp_step_size

        self.GM_IDN_MIN, self.GM_IDN_MAX = GM_IDN_RANGE
        self.GM_IDP_MIN, self.GM_IDP_MAX = GM_IDP_RANGE
        self.IBIAS_MIN,  self.IBIAS_MAX  = IBIAS_RANGE
        self.LN_LP_MIN,  self.LN_LP_MAX  = LN_LP_RANGE

        self.LP        = float(LP)
        self.Cin       = float(Cin)
        self.Fc_max     = float(Fc_max)

        self.area_budget_um2 = area_budget_um2
        self.area_weight     = area_weight

        self.vnin_target         = vnin_target
        self.vnin_reference      = vnin_reference
        self.vnin_gate_threshold = vnin_gate_threshold

        self.vnin_weight    = vnin_weight
        self.power_weight   = power_weight
        self.freq_weight    = freq_weight
        self.terminal_bonus = terminal_bonus
        self.ibias_ref      = ibias_ref
        self.fc_scale       = fc_scale

        self.mismatch_threshold = mismatch_threshold
        self.mode = int(mode)

        # -- Normalisation constants 
        self._gm_idn_mid   = (self.GM_IDN_MAX + self.GM_IDN_MIN) / 2.0
        self._gm_idn_half  = (self.GM_IDN_MAX - self.GM_IDN_MIN) / 2.0
        self._gm_idp_mid   = (self.GM_IDP_MAX + self.GM_IDP_MIN) / 2.0
        self._gm_idp_half  = (self.GM_IDP_MAX - self.GM_IDP_MIN) / 2.0
        self._ibias_mid    = (self.IBIAS_MAX  + self.IBIAS_MIN)  / 2.0
        self._ibias_half   = (self.IBIAS_MAX  - self.IBIAS_MIN)  / 2.0
        self._ln_lp_mid    = (self.LN_LP_MAX  + self.LN_LP_MIN)  / 2.0
        self._ln_lp_half   = (self.LN_LP_MAX  - self.LN_LP_MIN)  / 2.0
        self._vnin_mid     = vnin_gate_threshold
        self._vnin_half    = max(vnin_reference - vnin_gate_threshold, 1e-6)
        self._fc_mid       = Fc_max / 2.0
        self._fc_half      = Fc_max / 2.0
        self._area_mid     = area_budget_um2
        self._area_half    = area_budget_um2

        # Observations
        self.observation_space = spaces.Box(
            low=np.full(8, -2.0, dtype=np.float32),
            high=np.full(8,  2.0, dtype=np.float32),
            dtype=np.float32,
        )

        # Actions: [dGM_IDN, dGM_IDP, dIbias, dLN_LP_ratio]
        self.action_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4,  1.0, dtype=np.float32),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.state = None


        print("Circuit parameters")
        print(f"  LP (fixed)        : {self.LP:.2f} um")
        print(f"  LN_LP_RANGE       :  [{self.LN_LP_MIN}, {self.LN_LP_MAX}] um")
        print(f"  ln_lp_step_size   : {self.ln_lp_step_size} per step")
        print(f"  Cin               : {self.Cin*1e12:.1f} pF    Fc_max: {self.Fc_max/1e3:.0f} kHz")
        print(f"  GM_IDN_RANGE      : [{self.GM_IDN_MIN}, {self.GM_IDN_MAX}], GM_IDP_RANGE: [{self.GM_IDP_MIN}, {self.GM_IDP_MAX}]")
        print(f"  IBIAS_RANGE       : [{self.IBIAS_MIN}, {self.IBIAS_MAX}] uA)  " f"ibias_ref: {self.ibias_ref*1e6:.0f} uA")
        print()
        print("Area constraint")
        print(f"  area_budget_um2   : {self.area_budget_um2:.0f} um^2")
        print(f"  area_weight       : {self.area_weight}")
        print()
        print("Noise targets")
        print(f"  vnin_target       : {self.vnin_target} nV/rtHz  (terminal success)")
        print(f"  vnin_reference    : {self.vnin_reference} nV/rtHz  (reward normalisation)")
        print(f"  vnin_gate_thresh  : {self.vnin_gate_threshold} nV/rtHz  (fc reward gate)")
        print()
        print("Reward weights")
        print(f"  vnin_weight       : {self.vnin_weight}")
        print(f"  power_weight      : {self.power_weight}  (quadratic)")
        print(f"  freq_weight       : {self.freq_weight}   fc_scale={self.fc_scale/1e3:.0f} kHz")
        print(f"  area_weight       : {self.area_weight}  (linear above budget)")
        print(f"  terminal_bonus    : +{self.terminal_bonus}  (per step both targets met)")
        print(f"  mismatch_thresh   : {self.mismatch_threshold} S/A")
        print()
        
    def _compute_metrics(self, GM_IDN, GM_IDP, Ibias, LN_LP_ratio):
        """
        Update LNs on the amp_functions object (LP stays fixed),
        then call compute_all() once.
        """
        self.amp_functions.LNs = LN_LP_ratio * self.LP  
        self.amp_functions.LPs = self.LP

        return self.amp_functions.compute_all(
            GM_IDN, GM_IDP,
            Ibias=Ibias,
            Cin=self.Cin,
            Fc_max=self.Fc_max,
        )

    def _device_area_um2(self, m, LN_LP_ratio):
        """
        Input-pair area = WP*LP + WN*LN  (um^2) of one PMOS and NMOS transistor.
        """
        LP_um = self.LP                 
        LN_um = LN_LP_ratio * LP_um      
        return float(m['WP_um']) * LP_um + float(m['WN_um']) * LN_um

    def _normalise_obs(self, state, area_um2):
        GM_IDN, GM_IDP, VNin_nV, Fc, Ibias, LN_LP_ratio = state
        Fc_max_margin = (self.Fc_max - Fc) / self.Fc_max

        return np.array([
            np.clip((GM_IDN      - self._gm_idn_mid)  / self._gm_idn_half,  -2.0, 2.0),
            np.clip((GM_IDP      - self._gm_idp_mid)  / self._gm_idp_half,  -2.0, 2.0),
            np.clip((VNin_nV     - self._vnin_mid)     / self._vnin_half,    -2.0, 2.0),
            np.clip((Fc          - self._fc_mid)       / self._fc_half,      -2.0, 2.0),
            np.clip((Ibias       - self._ibias_mid)    / self._ibias_half,   -2.0, 2.0),
            np.clip(Fc_max_margin,                                             -2.0, 2.0),
            np.clip((LN_LP_ratio - self._ln_lp_mid)   / self._ln_lp_half,   -2.0, 2.0),
            np.clip((area_um2    - self._area_mid)     / self._area_half,    -2.0, 2.0),
        ], dtype=np.float32)

    def _mismatch_penalty(self, GM_IDN, GM_IDP):
        mismatch = abs(GM_IDN - GM_IDP)
        if mismatch <= self.mismatch_threshold:
            return 0.0
        excess = mismatch - self.mismatch_threshold
        return float(max(-2.0 * (np.exp(1.5 * excess) - 1.0), -5.0))

    def _area_penalty(self, area_um2):
        excess = max(0.0, area_um2 - self.area_budget_um2)
        return -self.area_weight * (excess / self.area_budget_um2)

    def _calculate_reward(self, VNin_nV, Fc, Ibias, area_um2):
        Fc_max_margin = self.Fc_max - Fc

        # Stage 1 (always active) 
        # Drive noise down with a power-law penalty, and a quadratic power penalty that scales in gradually as VNin improves
        r_vnin = -self.vnin_weight * (VNin_nV / self.vnin_reference) ** 1.5

        # progress [0, 1]: 0 at vnin_reference, 1 at vnin_gate_threshold
        progress = max(0.0, min(1.0,
            (self.vnin_reference - VNin_nV) /
            (self.vnin_reference - self.vnin_gate_threshold)
        ))
        r_power = -self.power_weight * (Ibias / self.ibias_ref) ** 2 * (0.15 + 0.85 * progress)

        # Stage 2 (gated: only once VNin < vnin_gate_threshold) 
        # Flicker corner reward not given to agent until the noise is pulled into a predefined range
        if VNin_nV < self.vnin_gate_threshold:
            r_fc   = self.freq_weight * float(np.tanh(Fc_max_margin / self.fc_scale))
            r_area = self._area_penalty(area_um2)
        else:
            r_fc   = 0.0
            r_area = 0.0

        reward = r_vnin + r_power + r_fc + r_area

        # Terminal (success) bonus: both Stage 1 and Stage 2 objectives satisfied
        if VNin_nV < self.vnin_target and Fc_max_margin > 0.0:
            reward += self.terminal_bonus

        return float(reward), False

    def step(self, action):
        GM_IDN, GM_IDP, VNin_nV, Fc, Ibias, LN_LP_ratio = self.state

        GM_IDN      = float(np.clip(GM_IDN      + action[0] * self.gm_step_size,
                                    self.GM_IDN_MIN, self.GM_IDN_MAX))
        GM_IDP      = float(np.clip(GM_IDP      + action[1] * self.gm_step_size,
                                    self.GM_IDP_MIN, self.GM_IDP_MAX))
        Ibias       = float(np.clip(Ibias       + action[2] * self.ibias_step_size,
                                    self.IBIAS_MIN, self.IBIAS_MAX))
        LN_LP_ratio = float(np.clip(LN_LP_ratio + action[3] * self.ln_lp_step_size,
                                    self.LN_LP_MIN, self.LN_LP_MAX))

        m        = self._compute_metrics(GM_IDN, GM_IDP, Ibias, LN_LP_ratio)
        VNin_nV  = m['VNin_nV']
        Fc       = m['Fc']
        Av0      = m['Av0']
        area_um2 = self._device_area_um2(m, LN_LP_ratio)

        self.state = np.array(
            [GM_IDN, GM_IDP, VNin_nV, Fc, Ibias, LN_LP_ratio],
            dtype=np.float32)

        reward, terminated = self._calculate_reward(VNin_nV, Fc, Ibias, area_um2)

        if self.mode in (0, 2):
            reward += self._mismatch_penalty(GM_IDN, GM_IDP)

        Fc_max_margin = self.Fc_max - Fc

        info = {
            'vn_in'         : VNin_nV,
            'fc'            : Fc,
            'Fc_max_margin'  : Fc_max_margin,
            'fc_ok'         : Fc_max_margin > 0.0,
            'av0'           : Av0,
            'power_uW'      : m['Power_uW'],
            'ibias'         : Ibias,
            'gm_idn'        : GM_IDN,
            'gm_idp'        : GM_IDP,
            'ln_lp_ratio'   : LN_LP_ratio,
            'LN_um'         : LN_LP_ratio * self.LP,   
            'area_um2'      : area_um2,
            'alpha'         : m['alpha'],
            'terminated'    : terminated,
            'GM_IDN_delta'  : action[0] * self.gm_step_size,
            'GM_IDP_delta'  : action[1] * self.gm_step_size,
            'Ibias_delta'   : action[2] * self.ibias_step_size,
            'LN_LP_delta'   : action[3] * self.ln_lp_step_size,
            'gain'          : Av0,
            'target_gain'   : 0.0,
            'meets_constraint': VNin_nV < self.vnin_target,
            'gain_violation': 0.0,
            'gain_excess'   : 0.0,
            'gain_error'    : 0.0,
        }

        return self._normalise_obs(self.state, area_um2), reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        def rand_inner(lo, hi):
            span = hi - lo
            return float(self.np_random.uniform(lo + 0.15 * span, hi - 0.15 * span))

        GM_IDN      = rand_inner(self.GM_IDN_MIN, self.GM_IDN_MAX)
        GM_IDP      = rand_inner(self.GM_IDP_MIN, self.GM_IDP_MAX)
        Ibias       = rand_inner(self.IBIAS_MIN,  self.IBIAS_MAX)
        LN_LP_ratio = rand_inner(self.LN_LP_MIN,  self.LN_LP_MAX)

        m        = self._compute_metrics(GM_IDN, GM_IDP, Ibias, LN_LP_ratio)
        VNin_nV  = m['VNin_nV']
        Fc       = m['Fc']
        area_um2 = self._device_area_um2(m, LN_LP_ratio)

        self.state = np.array(
            [GM_IDN, GM_IDP, VNin_nV, Fc, Ibias, LN_LP_ratio],
            dtype=np.float32)

        Fc_max_margin = self.Fc_max - Fc

        info = {
            'vn_in'          : VNin_nV,
            'fc'             : Fc,
            'Fc_max_margin'   : Fc_max_margin,
            'fc_ok'          : Fc_max_margin > 0.0,
            'av0'            : m['Av0'],
            'power_uW'       : m['Power_uW'],
            'ibias'          : Ibias,
            'gm_idn'         : GM_IDN,
            'gm_idp'         : GM_IDP,
            'ln_lp_ratio'    : LN_LP_ratio,
            'LN_um'          : LN_LP_ratio * self.LP,
            'area_um2'       : area_um2,
            'alpha'          : m['alpha'],
            'penalty'        : 0.0,
            'gain'           : m['Av0'],
            'target_gain'    : 0.0,
            'meets_constraint': VNin_nV < self.vnin_target,
            'gain_violation' : 0.0,
            'gain_excess'    : 0.0,
            'gain_error'     : 0.0,
            'GM_IDN_delta'   : 0,
            'GM_IDP_delta'   : 0,
            'Ibias_delta'    : 0,
            'LN_LP_delta'    : 0,
        }

        return self._normalise_obs(self.state, area_um2), info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
