"""
Training loop for the CCIA gm/ID RL optimiser.

Action space: [delta GM_IDN, delta GM_IDP, delta Ibias, delta LN_LP_ratio]  (4-dim)
Observation : [GM_IDN, GM_IDP, VNin, Fc, Ibias, Fc_max_margin, LN_LP, Area_norm] (8-dim)
"""

import json
import time
import os
import numpy as np
import torch


def train(
    # Classes to pass in 
    PPOAgent,
    EnviromentSetup,
    amp_functions,
    # Agent / network
    lr                    = 3e-4,
    gamma                 = 0.99,
    gae_lambda            = 0.95,
    clip_epsilon          = 0.2,
    value_coef            = 0.5,
    entropy_coef          = 0.10,
    max_grad_norm         = 0.5,
    hidden_dim            = 128,
    batch_size            = 128,
    epochs                = 4,
    device                = None,
    # Training schedule
    num_episodes          = 1000,
    num_val_episodes      = 20,
    max_steps_per_episode = 100,
    update_interval       = 10,
    validation_interval   = 50,
    # Early stopping
    patience              = 10,
    # Output directories 
    output_dir            = 'runs',
    checkpoint_dir        = None,
    best_model_dir        = None,
    episode_data_dir      = None,
    log_dir               = None,
    validation_log_dir    = None,
):
    checkpoint_dir     = checkpoint_dir     or os.path.join(output_dir, 'checkpoints')
    best_model_dir     = best_model_dir     or os.path.join(output_dir, 'best')
    episode_data_dir   = episode_data_dir   or os.path.join(output_dir, 'episodes')
    log_dir            = log_dir            or os.path.join(output_dir, 'logs')
    validation_log_dir = validation_log_dir or os.path.join(output_dir, 'validation')

    for d in [checkpoint_dir, best_model_dir, episode_data_dir,
              log_dir, validation_log_dir]:
        os.makedirs(d, exist_ok=True)

    best_model_path   = os.path.join(best_model_dir, 'best_validated_model.pth')
    final_model_path  = os.path.join(best_model_dir, 'final_model.pth')
    overall_best_path = os.path.join(best_model_dir, 'best_training_overall.pth')
    training_log_path = os.path.join(log_dir, 'training_log.json')
    episode_log_path  = os.path.join(log_dir, 'episode_log.json')

    # Device
    if device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = torch.device(device)

    print("=" * 70)
    print("CCIA gm/ID RL TRAINER  (with LN/LP ratio control)")
    print("=" * 70)
    print(f"Device              : {_device}")
    print(f"Output dir          : {output_dir}")
    print()
    print("Agent")
    print(f"  lr={lr}  hidden_dim={hidden_dim}  entropy_coef={entropy_coef}")
    print(f"  gamma={gamma}  gae_lambda={gae_lambda}  clip_eps={clip_epsilon}")
    print()
    print("Schedule")
    print(f"  num_episodes={num_episodes}  max_steps={max_steps_per_episode}")
    print(f"  update_interval={update_interval}  val_interval={validation_interval}")
    print(f"  patience={patience}")
    print("=" * 70)
    print()
    
    # Build environment 
    env = EnviromentSetup.VNinsEnv(
        amp_functions       = amp_functions,
        mode                = 0,
    )

    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.shape[0]         # 4
    print(f"Env ready  -- state_dim={state_dim}, action_dim={action_dim}\n")

    # Build agent
    agent = PPOAgent(
        state_dim     = state_dim,
        action_dim    = action_dim,
        lr            = lr,
        gamma         = gamma,
        gae_lambda    = gae_lambda,
        clip_epsilon  = clip_epsilon,
        value_coef    = value_coef,
        entropy_coef  = entropy_coef,
        max_grad_norm = max_grad_norm,
        epochs        = epochs,
        batch_size    = batch_size,
        hidden_dim    = hidden_dim,
        device        = _device,
    )

    # Training state
    best_reward          = float('-inf')
    best_val_reward      = float('-inf')
    no_improvement_count = 0
    best_vnin_seen       = float('inf')

    episode_rewards  = []
    episode_log      = []
    val_rewards      = []
    val_vnins        = []
    val_fcs          = []
    success_episodes = []

    start_time = time.time()

    for episode in range(num_episodes):

        state, info    = env.reset()
        episode_reward = 0.0
        step_log       = []
        terminated_ep  = False

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if isinstance(reward, np.ndarray):
                reward = reward.item()

            agent.store_transition(reward, done)
            episode_reward += reward

            step_log.append({
                'step'        : step + 1,
                'vn_in'       : round(float(info['vn_in']),         4),
                'fc'          : round(float(info['fc']),             2),
                'fc_ok'       : bool(info['fc_ok']),
                'ibias_uA'    : round(float(info['ibias']) * 1e6,    3),
                'power_uW'    : round(float(info['power_uW']),       3),
                'gm_idn'      : round(float(info['gm_idn']),         4),
                'gm_idp'      : round(float(info['gm_idp']),         4),
                'ln_lp_ratio' : round(float(info['ln_lp_ratio']),    4),
                'LN_um'       : round(float(info['LN_um']),          1),
                'area_um2'    : round(float(info['area_um2']),       2),
                'reward'      : round(float(reward),                 6),
                'terminated'  : bool(terminated),
            })

            state = next_state

            if done:
                terminated_ep = terminated
                break

        episode_rewards.append(episode_reward)

        final_vnin    = float(info['vn_in'])
        final_fc      = float(info['fc'])
        final_ibias   = float(info['ibias'])
        final_gmidn   = float(info['gm_idn'])
        final_gmidp   = float(info['gm_idp'])
        final_ln_lp   = float(info['ln_lp_ratio'])
        final_LN_um   = float(info['LN_um'])
        final_area    = float(info['area_um2'])
        fc_ok         = bool(info['fc_ok'])
        Fc_max_marg    = float(info['Fc_max_margin'])

        if final_vnin < best_vnin_seen:
            best_vnin_seen = final_vnin
            agent.save(os.path.join(best_model_dir, 'best_vnin.pth'))

        # Prints finished parameters every episode
        success_tag = "  *** SUCCESS ***" if terminated_ep else ""
        print(
            f"Ep {episode+1:>4} | "
            f"R: {episode_reward:>7.3f} | "
            f"VNin: {final_vnin:>5.2f} nV/rtHz | "
            f"Fc: {final_fc/1e3:>6.1f} kHz | "
            f"margin: {Fc_max_marg/1e3:>6.1f} kHz | "
            f"fc: {'Y' if fc_ok else 'N'} | "
            f"Ib: {final_ibias*1e6:>5.1f} uA | "
            f"N: {final_gmidn:>5.2f} | "
            f"P: {final_gmidp:>5.2f} | "
            f"LN/LP: {final_ln_lp:.2f} | "
            f"LN: {final_LN_um:.0f} um | "
            f"A: {final_area:.0f} um2 | "
        )

        # Record successful episodes 
        if terminated_ep:
            success_ep = {
                'episode'     : episode + 1,
                'reward'      : float(episode_reward),
                'steps'       : step + 1,
                'vn_in'       : final_vnin,
                'fc'          : final_fc,
                'ibias_uA'    : final_ibias * 1e6,
                'gm_idn'      : final_gmidn,
                'gm_idp'      : final_gmidp,
                'ln_lp_ratio' : final_ln_lp,
                'LN_um'       : final_LN_um,
                'area_um2'    : final_area,
            }
            success_episodes.append(success_ep)
            agent.save(os.path.join(best_model_dir, f'success_ep{episode+1}.pth'))
            agent.save(overall_best_path)

        elif episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(best_model_dir, f'best_reward_ep{episode+1}.pth'))

        # episode record for plotitng
        ep_record = {
            'episode'    : episode + 1,
            'reward'     : round(float(episode_reward), 4),
            'steps'      : step + 1,
            'terminated' : terminated_ep,
            'final': {
                'vn_in'       : round(final_vnin,          4),
                'fc'          : round(final_fc,             2),
                'fc_ok'       : fc_ok,
                'ibias_uA'    : round(final_ibias * 1e6,   3),
                'gm_idn'      : round(final_gmidn,         4),
                'gm_idp'      : round(final_gmidp,         4),
                'ln_lp_ratio' : round(final_ln_lp,         4),
                'LN_um'       : round(final_LN_um,         1),
                'area_um2'    : round(final_area,          2),
            },
            'steps_detail': step_log,
        }
        episode_log.append(ep_record)

        # Policy update
        if (episode + 1) % update_interval == 0 and len(agent.states) > 0:
            losses = agent.update(next_state)
            print(
                f"  [PPO update]  "
                f"policy={losses['policy_loss']:.4f}  "
                f"value={losses['value_loss']:.4f}  "
                f"entropy={losses['entropy_loss']:.4f}"
            )

        # Validation
        if (episode + 1) % validation_interval == 0:
            val_results = _test_policy(
                agent, env,
                num_episodes          = num_val_episodes,
                vnin_target           = env.vnin_target,
                max_steps_per_episode = max_steps_per_episode,
            )

            val_reward = val_results['mean_reward']
            val_rewards.append(val_reward)
            val_vnins.append(val_results['mean_vnin'])
            val_fcs.append(val_results['mean_fc'])

            print(
                f"\n  [Validation @ ep {episode+1}]  "
                f"mean_reward={val_reward:.3f} | "
                f"mean_VNin={val_results['mean_vnin']:.3f} nV/rtHz | "
                f"mean_Fc={val_results['mean_fc']/1e3:.1f} kHz | "
                f"mean_LN/LP={val_results['mean_ln_lp']:.3f} | "
                f"mean_area={val_results['mean_area']:.1f} um2 | "
                f"success={val_results['success_rate']*100:.1f}%\n"
            )

            val_snapshot = {
                'episode'      : episode + 1,
                'mean_reward'  : val_results['mean_reward'],
                'mean_vnin'    : val_results['mean_vnin'],
                'mean_fc'      : val_results['mean_fc'],
                'mean_ln_lp'   : val_results['mean_ln_lp'],
                'mean_area'    : val_results['mean_area'],
                'success_rate' : val_results['success_rate'],
                'all_vnins'    : [round(v, 4) for v in val_results['all_vnins']],
                'all_fcs'      : [round(v, 2) for v in val_results['all_fcs']],
                'all_ln_lps'   : [round(v, 4) for v in val_results['all_ln_lps']],
                'all_areas'    : [round(v, 2) for v in val_results['all_areas']],
            }
            with open(os.path.join(
                    validation_log_dir, f'validation_ep{episode+1}.json'), 'w') as f:
                json.dump(val_snapshot, f, indent=4)

            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{episode+1}.pth'))

            if val_reward > best_val_reward:
                best_val_reward      = val_reward
                no_improvement_count = 0
                agent.save(best_model_path)
                print(f"  ** New best validation model saved (reward={val_reward:.3f})")
            else:
                no_improvement_count += 1
                print(f"  No improvement ({no_improvement_count}/{patience})")

            if no_improvement_count >= patience:
                print(
                    f"\n  Early stopping at episode {episode+1} "
                    f"(no improvement for {patience} validations)"
                )
                break

    # Final save
    agent.save(final_model_path)
    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Episodes run          : {episode + 1}")
    print(f"  Total time            : {total_time/60:.1f} min  "
          f"({total_time/(episode+1):.2f} s/ep)")
    print(f"  Best training reward  : {best_reward:.4f}")
    print(f"  Best validation reward: {best_val_reward:.4f}")
    print(f"  Best VNin seen        : {best_vnin_seen:.4f} nV/rtHz")
    print(f"  Successful episodes   : {len(success_episodes)}")
    print("=" * 70)

    with open(episode_log_path, 'w') as f:
        json.dump(episode_log, f, indent=4)
    print(f"  Episode log  -> {episode_log_path}")

    log = {
        'config': {
            'LP'                  : env.LP,
            'Cin'                 : env.Cin,
            'Fc_max'               : env.Fc_max,
            'GM_IDN_RANGE'   : [env.GM_IDN_MIN, env.GM_IDN_MAX],
            'GM_IDP_RANGE'   : [env.GM_IDP_MIN, env.GM_IDP_MAX],
            'IBIAS_RANGE'    : [env.IBIAS_MIN,  env.IBIAS_MAX],
            'LN_LP_RANGE'    : [env.LN_LP_MIN,  env.LN_LP_MAX],
            'ibias_ref'           : env.ibias_ref,
            'area_budget_um2'     : env.area_budget_um2,
            'area_weight'         : env.area_weight,
            'vnin_target'         : env.vnin_target,
            'vnin_reference'      : env.vnin_reference,
            'vnin_gate_threshold' : env.vnin_gate_threshold,
            'vnin_weight'         : env.vnin_weight,
            'power_weight'        : env.power_weight,
            'freq_weight'         : env.freq_weight,
            'terminal_bonus'      : env.terminal_bonus,
            'mismatch_threshold'  : env.mismatch_threshold,
            'fc_scale'            : env.fc_scale,
            'gm_step_size'        : env.gm_step_size,
            'ibias_step_size'     : env.ibias_step_size,
            'ln_lp_step_size'     : env.ln_lp_step_size,
            'lr'                  : lr,
            'hidden_dim'          : hidden_dim,
            'entropy_coef'        : entropy_coef,
            'num_episodes'        : num_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'update_interval'     : update_interval,
            'validation_interval' : validation_interval,
            'patience'            : patience,
        },
        'results': {
            'best_training_reward'  : best_reward,
            'best_validation_reward': best_val_reward,
            'best_vnin_nV'          : best_vnin_seen,
            'success_count'         : len(success_episodes),
            'episodes_run'          : episode + 1,
            'total_time_s'          : total_time,
        },
        'episode_rewards'  : episode_rewards,
        'validation': {
            'rewards' : val_rewards,
            'vnins'   : val_vnins,
            'fcs'     : val_fcs,
        },
        'success_episodes' : success_episodes,
    }

    with open(training_log_path, 'w') as f:
        json.dump(log, f, indent=4)
    print(f"  Training log -> {training_log_path}")

    return agent, log


# Internal validation helper

def _test_policy(agent, env, num_episodes=20,
                 vnin_target=4.0, max_steps_per_episode=100):
    all_rewards = []
    all_vnins   = []
    all_fcs     = []
    all_ln_lps  = []
    all_areas   = []
    successes   = 0

    prev_mode = env.mode
    env.mode  = 2

    for _ in range(num_episodes):
        state, _       = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps_per_episode):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            if isinstance(reward, np.ndarray):
                reward = reward.item()
            episode_reward += reward
            if terminated or truncated:
                break

        all_rewards.append(episode_reward)
        all_vnins.append(float(info['vn_in']))
        all_fcs.append(float(info['fc']))
        all_ln_lps.append(float(info['ln_lp_ratio']))
        all_areas.append(float(info['area_um2']))

        if float(info['vn_in']) < vnin_target and bool(info['fc_ok']):
            successes += 1

    env.mode = prev_mode

    return {
        'mean_reward' : float(np.mean(all_rewards)),
        'mean_vnin'   : float(np.mean(all_vnins)),
        'mean_fc'     : float(np.mean(all_fcs)),
        'mean_ln_lp'  : float(np.mean(all_ln_lps)),
        'mean_area'   : float(np.mean(all_areas)),
        'min_vnin'    : float(np.min(all_vnins)),
        'success_rate': successes / num_episodes,
        'all_vnins'   : all_vnins,
        'all_fcs'     : all_fcs,
        'all_ln_lps'  : all_ln_lps,
        'all_areas'   : all_areas,
    }
