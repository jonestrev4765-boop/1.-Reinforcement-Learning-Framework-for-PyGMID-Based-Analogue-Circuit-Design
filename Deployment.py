"""

Deployment runner for a trained CCIA gm/ID PPO agent.

    runner = DeploymentRunner(
        model_path   = 'filepath/model.pth',
        amp_functions = amp_functions,          # Circuit Equations
        output_dir   = 'filepath/deploy',
    )
    
    runner.save_designs()
    writes CSV + JSON
"""

import os
import json
import csv
import time
from copy import deepcopy

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import AgentSetup
import EnviromentSetup


plt.rcParams.update({
    'figure.dpi'       : 120,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.size'        : 9,
})

_DESIGN_COLS = [
    'run', 'seed',
    'gm_idn', 'gm_idp', 'ibias_uA', 'ln_lp_ratio', 'LN_um',
    'vn_in_nV', 'fc_Hz', 'fc_ok',
    'area_um2', 'power_uW',
    'av0', 'alpha', 'reward', 'steps', 'success',
]

class DeploymentRunner:
    """
    Loads a trained PPO checkpoint and runs the agent in deployment mode
    (VNinsEnv mode=1 — penalties disabled)
    
    """

    def __init__(
        self,
        model_path,
        amp_functions,
        output_dir  = 'deployment',
        env_kwargs  = None,
        state_dim   = 8,
        action_dim  = 4,
        hidden_dim  = 128,
        device      = None,
    ):
        self.model_path    = model_path
        self.amp_functions = amp_functions
        self.output_dir    = output_dir
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.hidden_dim    = hidden_dim

        # Output directories
        self.plot_dir    = os.path.join(output_dir, 'plots')
        self.designs_dir = os.path.join(output_dir, 'designs')
        for d in [self.plot_dir, self.designs_dir]:
            os.makedirs(d, exist_ok=True)
            
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"[Deploy] Device : {self.device}")

        # Initialise agent 
        self.agent = AgentSetup.PPOAgent(
            state_dim  = state_dim,
            action_dim = action_dim,
            hidden_dim = hidden_dim,
            device     = self.device,
        )
        self.agent.load(model_path)
        self.agent.policy.eval()
        print(f"[Deploy] Model  : {model_path}")

        # build enviroment
        if env_kwargs:
            _env_kw = {}
            _env_kw.update(env_kwargs)

        self.env = EnviromentSetup.VNinsEnv(
            amp_functions = amp_functions,
            mode = 1, # Penalties off
            **_env_kw
        )
        print(f"[Deploy] Env    : state={state_dim}  action={action_dim}  mode=1 (deploy)\n")
        
        self.episode_records : list[dict] = []   # full step-by-step traces
        self.design_records  : list[dict] = []   # one row per episode (final design)
        self._run_complete   = False

    def run(
        self,
        n_episodes : int = 50,
        max_steps  : int = 100,
        seed_offset: int = 0,
        verbose    : bool = True,
    ) -> list[dict]:

        self.episode_records = []
        self.design_records  = []
        t0 = time.time()

        print(f"Running {n_episodes} deployment episodes …\n")

        for ep in range(n_episodes):
            seed = ep + seed_offset
            state, info = self.env.reset(seed=seed)

            ep_reward    = 0.0
            step_traces  = []

            for step in range(max_steps):
                with torch.no_grad():
                    action = self.agent.select_action(state, training=False)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += float(reward)
                state      = next_state

                step_traces.append({
                    'step'       : step + 1,
                    'vn_in'      : round(float(info['vn_in']),       4),
                    'fc'         : round(float(info['fc']),           2),
                    'fc_ok'      : bool(info['fc_ok']),
                    'ibias_uA'   : round(float(info['ibias']) * 1e6, 3),
                    'power_uW'   : round(float(info['power_uW']),    3),
                    'gm_idn'     : round(float(info['gm_idn']),      4),
                    'gm_idp'     : round(float(info['gm_idp']),      4),
                    'ln_lp_ratio': round(float(info['ln_lp_ratio']), 4),
                    'LN_um'      : round(float(info['LN_um']),       2),
                    'area_um2'   : round(float(info['area_um2']),    2),
                    'reward'     : round(float(reward),              6),
                })

                if terminated or truncated:
                    break

            success = (
                float(info['vn_in'])  < self.env.vnin_target
                and bool(info['fc_ok'])
            )

            design = {
                'run'         : ep + 1,
                'seed'        : seed,
                'gm_idn'      : round(float(info['gm_idn']),         4),
                'gm_idp'      : round(float(info['gm_idp']),         4),
                'ibias_uA'    : round(float(info['ibias']) * 1e6,    4),
                'ln_lp_ratio' : round(float(info['ln_lp_ratio']),    4),
                'LN_um'       : round(float(info['LN_um']),          3),
                'vn_in_nV'    : round(float(info['vn_in']),          4),
                'fc_Hz'       : round(float(info['fc']),             2),
                'fc_ok'       : bool(info['fc_ok']),
                'area_um2'    : round(float(info['area_um2']),       2),
                'power_uW'    : round(float(info['power_uW']),       3),
                'av0'         : round(float(info['av0']),            4),
                'alpha'       : round(float(info['alpha']),          5),
                'reward'      : round(ep_reward,                     4),
                'steps'       : step + 1,
                'success'     : success,
            }

            self.design_records.append(design)
            self.episode_records.append({
                'episode'    : ep + 1,
                'seed'       : seed,
                'reward'     : round(ep_reward, 4),
                'steps'      : step + 1,
                'success'    : success,
                'final'      : design,
                'step_traces': step_traces,
            })

            if verbose:
                tag = '  ✓ SUCCESS' if success else ''
                print(
                    f"Ep {ep+1:>3} | "
                    f"VNin={float(info['vn_in']):.3f} nV | "
                    f"Fc={float(info['fc'])/1e3:.1f} kHz | "
                    f"Ib={float(info['ibias'])*1e6:.1f} µA | "
                    f"A={float(info['area_um2']):.0f} µm² | "
                    f"R={ep_reward:.3f}{tag}"
                )

        elapsed     = time.time() - t0
        n_success   = sum(d['success'] for d in self.design_records)
        mean_vnin   = np.mean([d['vn_in_nV'] for d in self.design_records])
        best_vnin   = min(d['vn_in_nV'] for d in self.design_records)
        mean_area   = np.mean([d['area_um2'] for d in self.design_records])

        print(f"\n{'='*60}")
        print(f"  Episodes     : {n_episodes}")
        print(f"  Success rate : {n_success}/{n_episodes}  ({100*n_success/n_episodes:.1f}%)")
        print(f"  Mean VNin    : {mean_vnin:.3f} nV/√Hz")
        print(f"  Best VNin    : {best_vnin:.3f} nV/√Hz")
        print(f"  Mean Area    : {mean_area:.1f} µm²")
        print(f"  Wall time    : {elapsed:.1f} s")
        print(f"{'='*60}\n")

        self._run_complete = True
        return self.design_records

    # Save designs
    def save_designs(self, prefix: str = 'deployment') -> tuple[str, str]:
        """
        Persist all finalised designs as:
          • <designs_dir>/<prefix>_designs.csv   — flat table, one row per episode
          • <designs_dir>/<prefix>_designs.json  — full records incl. step traces
          • <designs_dir>/<prefix>_best.json     — single best design by VNin

        Returns (csv_path, json_path).
        """
        self._check_run()

        csv_path  = os.path.join(self.designs_dir, f'{prefix}_designs.csv')
        json_path = os.path.join(self.designs_dir, f'{prefix}_designs.json')
        best_path = os.path.join(self.designs_dir, f'{prefix}_best.json')

        # CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_DESIGN_COLS)
            writer.writeheader()
            writer.writerows(self.design_records)

        # JSON
        with open(json_path, 'w') as f:
            json.dump(self.episode_records, f, indent=2)

        # Best Design
        best = min(self.design_records, key=lambda d: d['vn_in_nV'])
        with open(best_path, 'w') as f:
            json.dump(best, f, indent=2)

        print(f"[Deploy] Designs saved:")
        print(f"         CSV  → {csv_path}")
        print(f"         JSON → {json_path}")
        print(f"         Best → {best_path}  (VNin={best['vn_in_nV']:.4f} nV/√Hz)")
        return csv_path, json_path
