import numpy as np
from utils.commons import load_data
import matplotlib.pyplot as plt
import os

def analyze_class_results(result_path, class_id):
    print(f"\nAnalyzing results for class {class_id}")
    results = load_data(result_path)

    Rs_pred = np.array(results['Rs_pred'])
    ts_pred = np.array(results['ts_pred'])
    Rs_gt = np.array(results['Rs_gt'])
    ts_gt = np.array(results['ts_gt'])
    chamfer_losses = np.array(results.get('chamfer_losses', []))

    R_errors = []
    for R_pred, R_gt in zip(Rs_pred, Rs_gt):
        error_rad = np.arccos(np.clip((np.trace(R_pred.T @ R_gt) - 1) / 2, -1, 1))
        error_deg = np.rad2deg(error_rad)
        R_errors.append(error_deg)

    t_errors = np.linalg.norm(ts_pred - ts_gt, axis=1) * 100

    print("\nRotation Error Statistics:")
    print(f"Mean: {np.mean(R_errors):.2f}°")
    print(f"Median: {np.median(R_errors):.2f}°")
    print(f"Std: {np.std(R_errors):.2f}°")

    print("\nTranslation Error Statistics:")
    print(f"Mean: {np.mean(t_errors):.2f} cm")
    print(f"Median: {np.median(t_errors):.2f} cm")
    print(f"Std: {np.std(t_errors):.2f} cm")

    if len(chamfer_losses) > 0:
        print("\nChamfer Loss Statistics:")
        print(f"Mean: {np.mean(chamfer_losses):.4f}")
        print(f"Median: {np.median(chamfer_losses):.4f}")
        print(f"Std: {np.std(chamfer_losses):.4f}")

    return R_errors, t_errors, chamfer_losses

def main():
    base_path = "./results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/eval_results"
    classes = ["000001", "000002", "000003"]

    all_R_errors = []
    all_t_errors = []
    all_chamfer_losses = []

    print("=== Detailed Results Analysis ===")

    for cls in classes:
        result_path = os.path.join(base_path, f"model_epoch19_T5_cosine_tudl_{cls}_noiseTrue_v1.pth")
        R_errors, t_errors, chamfer_losses = analyze_class_results(result_path, cls)
        all_R_errors.append(R_errors)
        all_t_errors.append(t_errors)
        if len(chamfer_losses) > 0:
            all_chamfer_losses.append(chamfer_losses)

    print("\n=== Overall Statistics ===")
    all_R_errors_flat = np.concatenate(all_R_errors)
    all_t_errors_flat = np.concatenate(all_t_errors)

    print("\nOverall Rotation Error:")
    print(f"Mean: {np.mean(all_R_errors_flat):.2f}°")
    print(f"Median: {np.median(all_R_errors_flat):.2f}°")

    print("\nOverall Translation Error:")
    print(f"Mean: {np.mean(all_t_errors_flat):.2f} cm")
    print(f"Median: {np.median(all_t_errors_flat):.2f} cm")

    if len(all_chamfer_losses) > 0:
        all_chamfer_losses_flat = np.concatenate(all_chamfer_losses)
        print("\nOverall Chamfer Loss:")
        print(f"Mean: {np.mean(all_chamfer_losses_flat):.4f}")
        print(f"Median: {np.median(all_chamfer_losses_flat):.4f}")

if __name__ == "__main__":
    main()
