import numpy as np
import pickle

from algorithms.evaluate import get_grasp_proxy


def load_archive(run_path):
    with open(run_path + "/data_export.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def evaluate_archive_grasp(run_path):
    data = load_archive(run_path)

    poses = data["success archive"]  # may need adjustment

    scores = []

    for pose in poses:
        score = get_grasp_proxy(pose)
        scores.append(score)

    return np.array(scores)


if __name__ == "__main__":
    run_path = "./runs/run13"  # CHANGE THIS
    scores = evaluate_archive_grasp(run_path)

    print("Mean grasp proxy:", np.mean(scores))
    print("Max grasp proxy:", np.max(scores))
