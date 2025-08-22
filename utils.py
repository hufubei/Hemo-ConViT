# src/utils.py

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score)
from torch.utils.data import DataLoader

# Import the model class for type hinting
from .model import ViTForRegressionContrastive


def load_model_from_h5(model_path: str, model_arch: str, projection_dim: int, device: torch.device) -> ViTForRegressionContrastive | None:
    """Loads a model's state_dict from an H5 file."""
    if not os.path.exists(model_path):
        print(f"Warning: Model file does not exist at {model_path}")
        return None
    try:
        # Instantiate a new model with the same architecture (weights are random)
        model = ViTForRegressionContrastive(
            model_name=model_arch,
            pretrained=False,
            projection_dim=projection_dim
        )
        # Load weights from the H5 file
        state_dict = {}
        with h5py.File(model_path, 'r') as f:
            for key in f.keys():
                state_dict[key] = torch.from_numpy(np.array(f[key][()]))
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def calculate_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_func, n_iterations=1000, seed=42):
    """Calculates a 95% confidence interval for a given metric using bootstrapping."""
    n_samples = len(y_true)
    if n_samples == 0: return (np.nan, np.nan, np.nan)
    
    rng = np.random.default_rng(seed=seed)
    metric_scores = []
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if len(indices) == 0: continue
        
        y_true_sample, y_pred_sample = y_true[indices], y_pred[indices]
        try:
            score = metric_func(y_true_sample, y_pred_sample)
            metric_scores.append(score)
        except ValueError:
            continue # Avoid errors if a bootstrap sample has only one class

    if not metric_scores:
        point_estimate = metric_func(y_true, y_pred)
        return (point_estimate, np.nan, np.nan)

    lower_bound = np.percentile(metric_scores, 2.5)
    upper_bound = np.percentile(metric_scores, 97.5)
    point_estimate = metric_func(y_true, y_pred)
    return point_estimate, lower_bound, upper_bound


def evaluate_model_for_training(model: nn.Module, data_loader: DataLoader, transform: nn.Sequential, device: torch.device):
    """Evaluates the model during training and returns R2 score and predictions."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img_raw, hgb_labels in data_loader:
            img_raw, hgb_labels = img_raw.to(device), hgb_labels.to(device)
            hgb_predictions = model(img_raw, apply_transform=transform)
            all_preds.extend(hgb_predictions.cpu().numpy())
            all_labels.extend(hgb_labels.cpu().numpy())
            
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    r2 = r2_score(all_labels_np, all_preds_np) if len(all_labels_np) > 1 else 0.0
    return r2, all_labels_np, all_preds_np


def final_evaluation_with_ci(model: nn.Module, data_loader: DataLoader, transform: nn.Sequential, device: torch.device, eval_name: str, bootstrap_iters: int):
    """Performs final evaluation and reports regression metrics with 95% CIs."""
    print(f"\n=== Final Regression Metrics with 95% CI on {eval_name.upper()} SET ===")
    model.eval()
    
    _, labels, preds = evaluate_model_for_training(model, data_loader, transform, device)
    
    if len(labels) < 10:
        print("Warning: Small dataset size. Bootstrap CIs may be unreliable.")

    mae, mae_low, mae_high = calculate_bootstrap_ci(labels, preds, mean_absolute_error, bootstrap_iters)
    rmse, rmse_low, rmse_high = calculate_bootstrap_ci(labels, preds, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), bootstrap_iters)
    r2, r2_low, r2_high = calculate_bootstrap_ci(labels, preds, r2_score, bootstrap_iters)

    print(f"MAE:  {mae:.3f} g/dL (95% CI: [{mae_low:.3f}, {mae_high:.3f}])")
    print(f"RMSE: {rmse:.3f} g/dL (95% CI: [{rmse_low:.3f}, {rmse_high:.3f}])")
    print(f"R²:   {r2:.3f} (95% CI: [{r2_low:.3f}, {r2_high:.3f}])")
    
    return labels, preds


def evaluate_as_classification(y_true_hgb: np.ndarray, y_pred_hgb: np.ndarray, threshold: float, plot_dir: str, dataset_name: str, bootstrap_iters: int):
    """Evaluates regression results as a binary classification task."""
    print(f"\n=== Binary Classification Evaluation on {dataset_name.upper()} SET (Threshold: {threshold} g/dL) ===")
    
    y_true_class = (y_true_hgb < threshold).astype(int)
    y_pred_class = (y_pred_hgb < threshold).astype(int)
    class_names = ['Non-Anemic', 'Anemic']
    
    accuracy, acc_low, acc_high = calculate_bootstrap_ci(y_true_class, y_pred_class, accuracy_score, bootstrap_iters)
    print(f"\nOverall Accuracy: {accuracy:.4f} (95% CI: [{acc_low:.4f}, {acc_high:.4f}])")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true_class, y_pred_class, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(y_true_class, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix on {dataset_name} Set (Hgb Threshold = {threshold} g/dL)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"confusion_matrix_{dataset_name.lower()}_thresh_{threshold}.png")
    plt.savefig(save_path)
    plt.show()


def plot_regression_performance(y_true_valid: np.ndarray, y_pred_valid: np.ndarray, y_true_test: np.ndarray, y_pred_test: np.ndarray, plot_dir: str):
    """Generates and saves a regression plot for both validation and test sets."""
    print("\n--- Generating Combined Regression Plot (Validation & Test Sets) ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(y_true_test, y_pred_test, alpha=0.6, s=50, color='steelblue', marker='o', label=f'Test Set (n={len(y_true_test)})')
    ax.scatter(y_true_valid, y_pred_valid, alpha=0.8, s=60, color='darkorange', marker='X', label=f'Validation Set (n={len(y_true_valid)})')
    
    all_vals = np.concatenate([y_true_valid, y_true_test])
    min_val, max_val = all_vals.min() - 0.5, all_vals.max() + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Prediction (y=x)')

    r2_valid = r2_score(y_true_valid, y_pred_valid)
    r2_test = r2_score(y_true_test, y_pred_test)
    text_str = f'$R^2$ (Validation): {r2_valid:.3f}\n$R^2$ (Test):    