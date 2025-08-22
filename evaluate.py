# src/evaluate.py

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import kornia.augmentation as K
import torch.nn as nn

# Import project modules
import config
from data_loader import AnemiaDataset
from utils import (load_model_from_h5, final_evaluation_with_ci,
                     evaluate_as_classification, plot_regression_performance,
                     plot_residual_analysis)

def main(args):
    """Main function to run the evaluation on a pre-trained model."""
    print(f"--- Starting Evaluation ---")
    print(f"Model Path: {args.model_path}")
    print(f"Anemia Threshold: {args.threshold} g/dL")
    print(f"Device: {config.DEVICE}")

    # Create a specific output directory for this evaluation run
    model_name = os.path.basename(args.model_path).split('.')[0]
    eval_plot_dir = os.path.join(config.PLOTS_OUTPUT_DIR, f"evaluation_{model_name}_thresh_{args.threshold}")
    os.makedirs(eval_plot_dir, exist_ok=True)
    print(f"Plots will be saved to: {eval_plot_dir}")

    # --- Load Model ---
    model = load_model_from_h5(
        model_path=args.model_path,
        model_arch=config.MODEL_ARCHITECTURE,
        projection_dim=config.PROJECTION_DIM,
        device=config.DEVICE
    )

    if not model:
        print("Evaluation stopped: could not load model.")
        return

    # --- GPU-based Evaluation Transform ---
    eval_transform_gpu = nn.Sequential(
        K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))
    ).to(config.DEVICE)
        
    # --- Load Data ---
    # We need to recreate the same data splits to ensure we evaluate on the correct data
    print("\nLoading and splitting dataset to get validation and test sets...")
    full_dataset = AnemiaDataset(
        excel_file=config.EXCEL_PATH,
        img_dir=config.IMG_DIR,
        image_size=config.IMAGE_SIZE,
        hgb_anemia_threshold=config.HGB_ANEMIA_THRESHOLD # Use original threshold for splitting
    )

    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    valid_size = int(config.VALID_RATIO * total_size)
    test_size = total_size - train_size - valid_size
    
    generator = torch.Generator().manual_seed(config.SEED)
    _, valid_dataset, test_dataset = random_split(
        full_dataset, [train_size, valid_size, test_size], generator=generator
    )

    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"Loaded {len(valid_dataset)} validation samples and {len(test_dataset)} test samples.")

    # --- Perform Evaluation ---
    # 1. Get regression metrics with confidence intervals
    labels_test, preds_test = final_evaluation_with_ci(model, test_loader, eval_transform_gpu, config.DEVICE, "TEST", config.BOOTSTRAP_ITERATIONS)
    labels_valid, preds_valid = final_evaluation_with_ci(model, valid_loader, eval_transform_gpu, config.DEVICE, "VALIDATION", config.BOOTSTRAP_ITERATIONS)

    # 2. Get classification metrics using the specified threshold
    evaluate_as_classification(labels_valid, preds_valid, args.threshold, eval_plot_dir, "Validation", config.BOOTSTRAP_ITERATIONS)
    evaluate_as_classification(labels_test, preds_test, args.threshold, eval_plot_dir, "Test", config.BOOTSTRAP_ITERATIONS)

    # 3. Generate and save plots
    plot_residual_analysis(labels_test, preds_test, eval_plot_dir)
    plot_regression_performance(labels_valid, preds_valid, labels_test, preds_test, eval_plot_dir)
    
    print(f"\n--- Evaluation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained HemaConViT regression model.")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint (.h5 file)."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=12.0, 
        help="The Hgb threshold in g/dL to use for binary classification (anemic vs. non-anemic)."
    )

    args = parser.parse_args()
    main(args)