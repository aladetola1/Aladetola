import numpy as np


def simulate_grasp_trial(sim, object_state):
    """
    Minimal stochastic grasp proxy.

    This replaces a missing physics-based grasp planner.
    Used for robustness estimation only.
    """

    pos = object_state["position"]
    ori = object_state["orientation"]

    # heuristic stability conditions
    height_ok = pos[2] > 0.02
    tilt_penalty = np.linalg.norm(ori)

    success_prob = 0.7 if height_ok and tilt_penalty < 1.0 else 0.2

    return np.random.rand() < success_prob


def evaluate_grasp_pose(sim, object_state, total_attempts=10,
                        pos_noise_std=0.005,
                        ori_noise_std=0.02):
    """
    Grasp-friendliness score via robustness under stochastic SE(3) perturbations.

    Computes expected grasp success under noisy object poses.
    """

    success_count = 0

    base_pos = object_state["position"]
    base_ori = object_state["orientation"]  # assumed quaternion [x, y, z, w]

    for _ in range(total_attempts):

        # --- position noise ---
        noisy_pos = base_pos + np.random.normal(0, pos_noise_std, size=3)

        # --- orientation noise (safe normalization) ---
        noisy_ori = base_ori + np.random.normal(0, ori_noise_std, size=4)
        noisy_ori = noisy_ori / np.linalg.norm(noisy_ori)  # IMPORTANT FIX

        noisy_state = {
            "position": noisy_pos,
            "orientation": noisy_ori
        }

        success = simulate_grasp_trial(sim, noisy_state)
        success_count += int(success)

    return success_count / total_attempts

def evaluate_robust_grasp(individual, env, n_samples=10):
    successes = []

    for i in range(n_samples):
        perturbed_env = apply_noise(env)
        success = run_grasp(perturbed_env, individual)
        successes.append(success)

    return np.mean(successes)

def batch_grasp_evaluation(sim, archive_states):
    scores = []

    for state in archive_states:
        score = evaluate_grasp_pose(sim, state)
        scores.append(score)

    return np.array(scores)
