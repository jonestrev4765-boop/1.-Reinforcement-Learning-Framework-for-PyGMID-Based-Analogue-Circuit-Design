"""

Plotting deployment results.

Loads a deployment JSON file produced by Deploy.py and generates all
figures. 

"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'figure.dpi'       : 120,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.size'        : 9,
})

# Default environment parameters 
_DEFAULT_ENV = dict(
    vnin_target         = 4.5,
    vnin_gate_threshold = 5.0,
    Fc_max              = 50_000,
    area_budget_um2     = 500,
)


def _rolling(arr, w):
    """ rolling mean"""
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        out[i] = arr[lo:i + 1].mean()
    return out


def find_best_episode(episode_records, design_records, env):
    def _full_success(d):
        return (
            d['vn_in_nV']  < env['vnin_target']
            and d['fc_ok']
            and d['area_um2'] <= env['area_budget_um2']
        )

    full = [ep for ep, d in zip(episode_records, design_records) if _full_success(d)]
    if full:
        best = min(full, key=lambda ep: ep['final']['ibias_uA'])
        label = 'full success (VNin ✓  Fc ✓  Area ✓)'
        return best, label

    vnin_fc = [
        ep for ep, d in zip(episode_records, design_records)
        if d['vn_in_nV'] < env['vnin_target'] and d['fc_ok']
    ]
    if vnin_fc:
        best = min(vnin_fc, key=lambda ep: ep['final']['ibias_uA'])
        label = 'best VNin+Fc (area exceeded)'
        return best, label

    vnin_only = [
        ep for ep, d in zip(episode_records, design_records)
        if d['vn_in_nV'] < env['vnin_target']
    ]
    if vnin_only:
        best = min(vnin_only, key=lambda ep: ep['final']['ibias_uA'])
        label = 'best VNin only'
        return best, label

    best = min(episode_records, key=lambda ep: ep['final']['vn_in_nV'])
    label = 'lowest VNin (no constraint met)'
    return best, label


class DeploymentPlotter:

    def __init__(self, json_path: str, env_params: dict = None):
        self.json_path = json_path

        # Define output directory
        data_dir      = os.path.dirname(os.path.abspath(json_path))
        parent_dir    = os.path.dirname(data_dir)
        self.plot_dir = os.path.join(parent_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        # environment parameters
        self.env = dict(_DEFAULT_ENV)
        if env_params:
            self.env.update(env_params)

        # load data
        with open(json_path) as f:
            self.episode_records = json.load(f)

        self.design_records = [ep['final'] for ep in self.episode_records]
        
        for ep, d in zip(self.episode_records, self.design_records):
            d.setdefault('run',     ep['episode'])
            d.setdefault('reward',  ep['reward'])
            d.setdefault('steps',   ep['steps'])
            d.setdefault('success', ep['success'])

        self.best_ep, self.best_label = find_best_episode(
            self.episode_records, self.design_records, self.env
        )

        n = len(self.design_records)
        print(f"[DeployPlots] Loaded {n} episodes from {json_path}")
        print(f"[DeployPlots] Best episode: #{self.best_ep['episode']}  ({self.best_label})")
        print(f"[DeployPlots] Plot dir    : {self.plot_dir}\n")

    def _d(self, key):
        return np.array([r[key] for r in self.design_records])

    def _w(self):
        return min(10, len(self.design_records) // 5 + 1)

    def _save(self, fig, name, prefix):
        path = os.path.join(self.plot_dir, f'{prefix}_{name}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"[DeployPlots] Saved → {path}")
        return path

    def plot_all(self, prefix: str = 'deployment', show: bool = False) -> dict:
        """Generate and save every figure.  Returns {name: fig}."""
        builders = {
            'reward_and_steps'   : self._fig_reward_steps,
            'noise_and_fc'       : self._fig_noise_fc,
            'design_space'       : self._fig_design_space,
            'parameter_evolution': self._fig_param_evolution,
            'within_episode'     : self._fig_within_episode,
            'constraint_summary' : self._fig_constraint_summary,
            'pareto_front'       : self._fig_pareto,
        }
        figs = {}
        for name, fn in builders.items():
            fig = fn()
            self._save(fig, name, prefix)
            figs[name] = fig
            if show:
                plt.show()
            else:
                plt.close(fig)

        print(f"\n[DeployPlots] All {len(figs)} plots saved to {self.plot_dir}\n")
        return figs

    # ── public: single episode plot ───────────────────────────────────────────

    def plot_episode(
        self,
        episode_number: int,
        prefix: str = 'deployment',
        show: bool = False,
    ):
        ep = next(
            (e for e in self.episode_records if e['episode'] == episode_number),
            None,
        )
        if ep is None:
            raise ValueError(
                f"Episode {episode_number} not found. "
                f"Valid range: 1–{len(self.episode_records)}"
            )
        fig = self._fig_episode_traces(ep)
        name = f'episode_{episode_number:03d}'
        self._save(fig, name, prefix)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # 1 ── Reward & steps
    def _fig_reward_steps(self):
        runs    = self._d('run')
        rewards = self._d('reward')
        steps   = self._d('steps')
        w       = self._w()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.plot(runs, rewards, alpha=0.35, color='steelblue', lw=0.8)
        ax.plot(runs, _rolling(rewards, w), color='steelblue', lw=2,
                label=f'Rolling mean ({w})')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total reward')
        ax.set_title('Episode Reward')
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.bar(runs, steps, color='slategray', alpha=0.6, width=0.8)
        ax.plot(runs, _rolling(steps, w), color='black', lw=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Steps to Termination')

        fig.suptitle('Deployment — Reward & Steps', fontweight='bold')
        fig.tight_layout()
        return fig

    # 2 ── Noise & Fc
    def _fig_noise_fc(self):
        runs    = self._d('run')
        vnins   = self._d('vn_in_nV')
        fcs     = self._d('fc_Hz') / 1e3
        success = self._d('success').astype(bool)
        w       = self._w()

        area_ok      = self._d('area_um2') <= self.env['area_budget_um2']
        full_success = success & area_ok

        fig, axes = plt.subplots(1, 2, figsize=(7, 4))

        ax = axes[0]
        ax.scatter(runs[~full_success], vnins[~full_success],
                   c='steelblue', s=20, alpha=0.6, label='Fail', zorder=3)
        ax.scatter(runs[full_success], vnins[full_success],
                   c='green', s=40, marker='*', alpha=0.9, label='Success', zorder=4)
        ax.axhline(self.env['vnin_target'],
                   color='green', ls='--', lw=1.2,
                   label=f"Target ({self.env['vnin_target']} nV/√Hz)")
        ax.set_xlabel('Episode')
        ax.set_ylabel('VNin (nV/√Hz)')
        ax.set_title('Input-Referred Noise')
        ax.legend(fontsize=8)

        ax = axes[1]
        Fc_kHz = self.env['Fc_max'] / 1e3
        fc_full_success = self._d('fc_ok').astype(bool) & area_ok
        ax.scatter(runs[~fc_full_success], fcs[~fc_full_success],
                   c='purple', s=20, alpha=0.6, label='Fail', zorder=3)
        ax.scatter(runs[fc_full_success], fcs[fc_full_success],
                   c='green', s=40, marker='*', alpha=0.9, label='Success', zorder=4)
        ax.axhline(Fc_kHz, color='red', ls='--', lw=1.2,
                   label=f'Fc limit = {Fc_kHz:.1f} kHz')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Fc (kHz)')
        ax.set_title('Flicker Corner Frequency')
        ax.legend(fontsize=8)

        fig.suptitle('Deployment — Noise & Frequency', fontweight='bold')
        fig.tight_layout()
        return fig

    # 3 ── Design space scatter matrix
    def _fig_design_space(self):
        keys    = ['gm_idn', 'gm_idp', 'ibias_uA', 'ln_lp_ratio', 'vn_in_nV', 'area_um2']
        labels  = ['gm/ID_N', 'gm/ID_P', 'Ibias (µA)', 'LN/LP', 'VNin (nV)', 'Area (µm²)']
        data    = np.column_stack([self._d(k) for k in keys])
        n       = len(keys)
        success = self._d('success').astype(bool)

        fig, axes = plt.subplots(n, n, figsize=(12, 12))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    ax.hist(data[:, i], bins=20, color='steelblue', alpha=0.7)
                    ax.set_xlabel(labels[i], fontsize=7)
                else:
                    ax.scatter(data[~success, j], data[~success, i],
                               c='steelblue', s=8, alpha=0.5)
                    ax.scatter(data[success, j], data[success, i],
                               c='green', s=18, alpha=0.9, marker='*')
                ax.tick_params(labelsize=6)
                if j == 0:
                    ax.set_ylabel(labels[i], fontsize=7)
                if i == n - 1:
                    ax.set_xlabel(labels[j], fontsize=7)

        fig.suptitle('Design Space Scatter Matrix  (★ = success)', fontweight='bold')
        fig.tight_layout()
        return fig

    # 4 ── Parameter evolution (early vs. late box-plots)
    def _fig_param_evolution(self):
        params = ['vn_in_nV', 'fc_Hz', 'ibias_uA', 'area_um2', 'reward']
        titles = ['VNin (nV/√Hz)', 'Fc (Hz)', 'Ibias (µA)', 'Area (µm²)', 'Reward']

        n   = len(self.design_records)
        mid = n // 2

        fig, axes = plt.subplots(1, len(params), figsize=(15, 4))
        for ax, p, t in zip(axes, params, titles):
            e_vals = [r[p] for r in self.design_records[:mid]]
            l_vals = [r[p] for r in self.design_records[mid:]]
            bp = ax.boxplot(
                [e_vals, l_vals],
                labels=['Early', 'Late'],
                patch_artist=True,
                medianprops=dict(color='red', lw=2),
            )
            bp['boxes'][0].set_facecolor('lightsteelblue')
            bp['boxes'][1].set_facecolor('lightsalmon')
            ax.set_title(t, fontsize=9)
            ax.tick_params(labelsize=8)

        fig.suptitle('Early vs. Late Episodes — Design Parameter Evolution', fontweight='bold')
        fig.tight_layout()
        return fig

    # 5 ── Within-episode traces (best episode by default)
    def _fig_within_episode(self):
        return self._fig_episode_traces(self.best_ep, label=self.best_label)

    def _fig_episode_traces(self, ep_record, label: str = None):
        traces  = ep_record['step_traces']
        steps   = [s['step']        for s in traces]
        vnins   = [s['vn_in']       for s in traces]
        fcs     = [s['fc'] / 1e3    for s in traces]
        ibias   = [s['ibias_uA']    for s in traces]
        area    = [s['area_um2']    for s in traces]
        power   = [s['power_uW']    for s in traces]
        rewards = [s['reward']      for s in traces]
        gmidn   = [s['gm_idn']      for s in traces]
        gmidp   = [s['gm_idp']      for s in traces]
        lnlp    = [s['ln_lp_ratio'] for s in traces]
        ln_um   = [s['LN_um']       for s in traces]
 
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        axes = axes.flatten()
 
        def _plot(ax, x, y, ylabel, color, hline=None, hline_label=None):
            ax.plot(x, y, color=color, lw=1.8)
            if hline is not None:
                ax.axhline(hline, color='red', ls='--', lw=1, label=hline_label)
                ax.legend(fontsize=7)
            ax.set_xlabel('Step')
            ax.set_ylabel(ylabel)
 
        _plot(axes[0], steps, vnins,   'VNin (nV/√Hz)', 'crimson',
              self.env['vnin_target'], f"Target {self.env['vnin_target']}")
        _plot(axes[1], steps, fcs,     'Fc (kHz)',       'purple',
              self.env['Fc_max'] / 1e3, 'Fc limit')
        _plot(axes[2], steps, area,    'Area (µm²)',     'sienna',
              self.env['area_budget_um2'], 'Budget')
        _plot(axes[3], steps, power,   'Power (µW)',     'darkorchid')
        
        _plot(axes[4], steps, gmidn,   'gm/ID_N (S/A)',  'royalblue')
        _plot(axes[5], steps, gmidp,   'gm/ID_P (S/A)',  'darkorange')
        _plot(axes[6], steps, ibias,   'Ibias (µA)',      'teal')
        _plot(axes[7], steps, lnlp,    'LN/LP ratio',     'olivedrab')
 
        _plot(axes[8],  steps, ln_um,   'LN (µm)',        'steelblue')
        _plot(axes[9],  steps, rewards, 'Step reward',    'dimgray')
 
        for idx in [10, 11]:
            axes[idx].axis('off')
 
        d = ep_record['final']
        summary = (
            f"Episode : #{ep_record['episode']}\n"
            + (f"({label})\n" if label else "")
            + f"\n── Outputs ──────────────────\n"
            f"VNin  = {d['vn_in_nV']:.4f} nV/√Hz\n"
            f"Fc    = {d['fc_Hz']/1e3:.2f} kHz\n"
            f"Area  = {d['area_um2']:.1f} µm²\n"
            f"Power = {d['power_uW']:.2f} µW\n"
            f"\n── Actions ──────────────────\n"
            f"gm/IDn = {d['gm_idn']:.3f} S/A\n"
            f"gm/IDp = {d['gm_idp']:.3f} S/A\n"
            f"Ibias  = {d['ibias_uA']:.2f} µA\n"
            f"LN/LP  = {d['ln_lp_ratio']:.3f}  LN = {d['LN_um']:.2f} µm\n"
            f"\n── Episode ──────────────────\n"
            f"Steps   = {ep_record['steps']}\n"
            f"Reward  = {ep_record['reward']:.4f}\n"
            f"Success : {'✓' if d['success'] else '✗'}"
        )
        
        axes[10].text(0.05, 0.97, summary, transform=axes[10].transAxes,
                      va='top', fontsize=9, family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
 
        ep_num = ep_record['episode']
        title  = f'Episode #{ep_num} — Step-by-Step Trajectory'
        if label:
            title += f'  [{label}]'
        fig.suptitle(title, fontweight='bold')
        fig.tight_layout()
        return fig

    # 6 ── Constraint summary bar chart
    def _fig_constraint_summary(self):
        n        = len(self.design_records)
        vnin_ok  = sum(1 for d in self.design_records if d['vn_in_nV'] < self.env['vnin_target'])
        fc_ok    = sum(1 for d in self.design_records if d['fc_ok'])
        area_ok  = sum(1 for d in self.design_records if d['area_um2'] <= self.env['area_budget_um2'])
        full_ok  = sum(
            1 for d in self.design_records
            if d['vn_in_nV'] < self.env['vnin_target']
            and d['fc_ok']
            and d['area_um2'] <= self.env['area_budget_um2']
        )
        success  = sum(1 for d in self.design_records if d['success'])

        labels = [
            f"VNin < {self.env['vnin_target']} nV",
            'Fc < limit',
            f"Area ≤ {self.env['area_budget_um2']} µm²",
            'All constraints',
            'Flagged success',
        ]
        counts = [vnin_ok, fc_ok, area_ok, full_ok, success]
        rates  = [c / n * 100 for c in counts]
        colors = ['#2196F3', '#9C27B0', '#FF9800', '#4CAF50', '#607D8B']

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        bars = ax.barh(labels, rates, color=colors, alpha=0.85, edgecolor='white')
        for bar, r, c in zip(bars, rates, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{r:.1f}%  ({c}/{n})', va='center', fontsize=8)
        ax.set_xlim(0, 115)
        ax.set_xlabel('Satisfaction rate (%)')
        ax.set_title('Constraint Satisfaction')
        ax.axvline(100, color='green', ls='--', lw=1)

        ax = axes[1]
        vnins = [d['vn_in_nV'] for d in self.design_records]
        vp = ax.violinplot([vnins], positions=[1], showmedians=True)
        for body in vp['bodies']:
            body.set_facecolor('steelblue')
            body.set_alpha(0.7)
        ax.axhline(self.env['vnin_target'],
                   color='green', ls='--', lw=1.2,
                   label=f"Target {self.env['vnin_target']} nV")
        ax.axhline(self.env['vnin_gate_threshold'],
                   color='orange', ls=':', lw=1.2,
                   label=f"Gate {self.env['vnin_gate_threshold']} nV")
        ax.set_xticks([1])
        ax.set_xticklabels(['VNin (nV/√Hz)'])
        ax.set_title('Noise Distribution')
        ax.legend(fontsize=8)

        fig.suptitle('Deployment — Constraint Satisfaction Summary', fontweight='bold')
        fig.tight_layout()
        return fig

    # 7 ── Pareto front
    def _fig_pareto(self):
        vnins   = self._d('vn_in_nV')
        powers  = self._d('power_uW')
        areas   = self._d('area_um2')
        success = self._d('success').astype(bool)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        sc = ax.scatter(powers[~success], vnins[~success], c=areas[~success],
                        cmap='viridis', s=25, alpha=0.6,
                        vmin=areas.min(), vmax=areas.max())
        ax.scatter(powers[success], vnins[success], c=areas[success],
                   cmap='viridis', s=80, marker='*', edgecolors='black', lw=0.5,
                   vmin=areas.min(), vmax=areas.max())
        plt.colorbar(sc, ax=ax, label='Area (µm²)')
        ax.axhline(self.env['vnin_target'], color='green', ls='--', lw=1.2,
                   label=f"VNin target {self.env['vnin_target']} nV")
        ax.set_xlabel('Power (µW)')
        ax.set_ylabel('VNin (nV/√Hz)')
        ax.set_title('VNin vs. Power  (colour = area)')
        ax.legend(fontsize=8)

        ax = axes[1]
        sc2 = ax.scatter(areas[~success], vnins[~success], c=powers[~success],
                         cmap='plasma', s=25, alpha=0.6)
        ax.scatter(areas[success], vnins[success], c=powers[success],
                   cmap='plasma', s=80, marker='*', edgecolors='black', lw=0.5)
        plt.colorbar(sc2, ax=ax, label='Power (µW)')
        ax.axhline(self.env['vnin_target'], color='green', ls='--', lw=1.2)
        ax.axvline(self.env['area_budget_um2'], color='orange', ls=':', lw=1.2,
                   label=f"Area budget {self.env['area_budget_um2']} µm²")
        ax.set_xlabel('Area (µm²)')
        ax.set_ylabel('VNin (nV/√Hz)')
        ax.set_title('VNin vs. Area  (colour = power)')
        ax.legend(fontsize=8)

        fig.suptitle('Deployment — Design Trade-off (Pareto) View  (★ = success)',
                     fontweight='bold')
        fig.tight_layout()
        return fig
