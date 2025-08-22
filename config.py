# config.py
import os
import torch

# --- Paths ---
# Usamos una variable de entorno para la base, o un valor por defecto.
DRIVE_BASE_PATH = os.getenv("DRIVE_BASE_PATH", "./") 
EXCEL_PATH = os.path.join(DRIVE_BASE_PATH, "AnemiaEyeDefyParaRegresion/total_defy.xlsx")
IMG_DIR = os.path.join(DRIVE_BASE_PATH, "AnemiaEyeDefyParaRegresion/Defy_total")
MODEL_SAVE_DIR = os.path.join(DRIVE_BASE_PATH, "model_checkpoints_regression_adaptive_contrastive_ROI")
PLOTS_OUTPUT_DIR = os.path.join(DRIVE_BASE_PATH, "model_metrics_plots_regression_adaptive_contrastive_ROI")

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reproducibility ---
SEED = 45

# --- Model & Training Hyperparameters ---
MODEL_ARCHITECTURE = 'deit_small_patch16_224'
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 52
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 2

# --- Data & Thresholds ---
HGB_ANEMIA_THRESHOLD = 11.5
TRAIN_RATIO = 0.86
VALID_RATIO = 0.07
TEST_RATIO = 0.07

# --- Contrastive Learning Parameters ---
CONTRASTIVE_LAMBDA = 0.4
PROJECTION_DIM = 128
TEMPERATURE = 0.1
HGB_SIMILARITY_SIGMA = 1.0

# --- Evaluation Parameters ---
BOOTSTRAP_ITERATIONS = 100