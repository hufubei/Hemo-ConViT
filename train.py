# src/train.py

import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import kornia.augmentation as K

# Import project modules
import config
from data_loader import AnemiaDataset, create_augmented_and_balanced_dataset
from model import ViTForRegressionContrastive, AdaptiveContrastiveLoss
from utils import (evaluate_model_for_training, final_evaluation_with_ci, 
                     evaluate_as_classification, plot_regression_performance, 
                     plot_residual_analysis, load_model_from_h5)
# Note: You will need to add a 'plot_training_history' function to utils.py for the final plots


def set_seeds(seed_value: int):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main function to run the training and evaluation pipeline."""
    set_seeds(config.SEED)
    print(f"Using device: {config.DEVICE}")

    # Ensure output directories exist
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_OUTPUT_DIR, exist_ok=True)
    
    # --- GPU-based Kornia Transforms ---
    train_transform_gpu = nn.Sequential(
        K.RandomRotation(degrees=15.0, p=0.5),
        K.RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.5),
        K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))
    ).to(config.DEVICE)
    
    eval_transform_gpu = nn.Sequential(
        K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))
    ).to(config.DEVICE)

    # --- 1. Data Preparation ---
    full_dataset = AnemiaDataset(
        excel_file=config.EXCEL_PATH,
        img_dir=config.IMG_DIR,
        image_size=config.IMAGE_SIZE,
        hgb_anemia_threshold=config.HGB_ANEMIA_THRESHOLD
    )

    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    valid_size = int(config.VALID_RATIO * total_size)
    test_size = total_size - train_size - valid_size

    print(f"\nSplitting dataset ({total_size} samples):")
    print(f"  - Training:   {train_size} ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"  - Validation: {valid_size} ({config.VALID_RATIO*100:.0f}%)")
    print(f"  - Test:       {test_size} ({config.TEST_RATIO*100:.0f}%)")

    generator = torch.Generator().manual_seed(config.SEED)
    train_original_subset, valid_dataset, test_dataset = random_split(
        full_dataset, [train_size, valid_size, test_size], generator=generator
    )
    
    train_final_dataset = create_augmented_and_balanced_dataset(
        train_original_subset, config.HGB_ANEMIA_THRESHOLD
    )

    train_loader = DataLoader(train_final_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # --- 2. Model, Loss, and Optimizer Setup ---
    model = ViTForRegressionContrastive(
        model_name=config.MODEL_ARCHITECTURE,
        pretrained=True,
        projection_dim=config.PROJECTION_DIM
    ).to(config.DEVICE)
    print(f"\nModel initialized: {config.MODEL_ARCHITECTURE}")

    criterion_regression = nn.SmoothL1Loss()
    criterion_contrastive = AdaptiveContrastiveLoss(
        temperature=config.TEMPERATURE, 
        hgb_similarity_sigma=config.HGB_SIMILARITY_SIGMA
    ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS)
    
    def warmup_lr(epoch):
        return (epoch + 1) / config.WARMUP_EPOCHS if epoch < config.WARMUP_EPOCHS else 1.0
        
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    scaler = GradScaler()

    # --- 3. Training Loop ---
    print("\n=== Starting Training ===")
    last_saved_model_path = None

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss_reg, running_loss_con = 0.0, 0.0
        optimizer.zero_grad()
        
        for i, (images_raw, hgb_labels) in enumerate(train_loader):
            images_raw, hgb_labels = images_raw.to(config.DEVICE), hgb_labels.to(config.DEVICE)
            
            with autocast():
                hgb_preds, features = model(images_raw, apply_transform=train_transform_gpu)
                loss_reg = criterion_regression(hgb_preds, hgb_labels)
                loss_con = criterion_contrastive(features, hgb_labels)
                loss_total = loss_reg + config.CONTRASTIVE_LAMBDA * loss_con
            
            loss_total_acc = loss_total / config.GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss_total_acc).backward()

            if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss_reg += loss_reg.item() * images_raw.size(0)
            running_loss_con += loss_con.item() * images_raw.size(0)

        # Step schedulers
        if epoch < config.WARMUP_EPOCHS:
            scheduler.step()
        else:
            cosine_scheduler.step()
        
        # Evaluate performance for the epoch
        valid_r2, _, _ = evaluate_model_for_training(model, valid_loader, eval_transform_gpu, config.DEVICE)
        test_r2, _, _ = evaluate_model_for_training(model, test_loader, eval_transform_gpu, config.DEVICE)
        
        print(f"Epoch {epoch+1:2d}/{config.NUM_EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.2e}) | "
              f"Valid R²: {valid_r2:.4f} | Test R²: {test_r2:.4f}", end='')

        # Save model checkpoint periodically
        if (epoch + 1) >= 10 and (epoch + 1) <= 45 and (epoch + 1) % 5 == 0:
            model_save_path = os.path.join(config.MODEL_SAVE_DIR, f"model_epoch_{epoch+1}.h5")
            last_saved_model_path = model_save_path
            print(" -> Saving model...")
            try:
                with h5py.File(model_save_path, 'w') as f:
                    for key, value in model.state_dict().items():
                        f.create_dataset(key, data=value.cpu().numpy())
            except Exception as e:
                print(f"  -> ERROR saving model: {e}")
        else:
            print()

    # --- 4. Final Evaluation ---
    print("\n--- Training Finished ---")
    
    if not last_saved_model_path:
        print("Could not perform final evaluation because no model was saved.")
        return

    print(f"\n--- Loading last saved model ({os.path.basename(last_saved_model_path)}) for final evaluation ---")
    final_model = load_model_from_h5(
        model_path=last_saved_model_path,
        model_arch=config.MODEL_ARCHITECTURE,
        projection_dim=config.PROJECTION_DIM,
        device=config.DEVICE
    )

    if final_model:
        # Perform regression evaluation with confidence intervals
        labels_test, preds_test = final_evaluation_with_ci(final_model, test_loader, eval_transform_gpu, config.DEVICE, "TEST", config.BOOTSTRAP_ITERATIONS)
        labels_valid, preds_valid = final_evaluation_with_ci(final_model, valid_loader, eval_transform_gpu, config.DEVICE, "VALIDATION", config.BOOTSTRAP_ITERATIONS)
        
        # Perform classification evaluation
        evaluate_as_classification(labels_valid, preds_valid, config.HGB_ANEMIA_THRESHOLD, config.PLOTS_OUTPUT_DIR, "Validation", config.BOOTSTRAP_ITERATIONS)
        evaluate_as_classification(labels_test, preds_test, config.HGB_ANEMIA_THRESHOLD, config.PLOTS_OUTPUT_DIR, "Test", config.BOOTSTRAP_ITERATIONS)

        # Generate final plots
        plot_residual_analysis(labels_test, preds_test, config.PLOTS_OUTPUT_DIR)
        plot_regression_performance(labels_valid, preds_valid, labels_test, preds_test, config.PLOTS_OUTPUT_DIR)
    else:
        print("\nCould not load the final model for evaluation.")


if __name__ == '__main__':
    main()