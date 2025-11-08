import numpy as np

def calculate_mpjpe(pred_coords, gt_coords):
    pred = np.array(pred_coords)
    gt = np.array(gt_coords)
    distances = np.linalg.norm(pred - gt, axis=1)
    return np.mean(distances)