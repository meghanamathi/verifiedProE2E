"""
Configuration file for Product Fraud Detection System
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
ORIGINAL_PRODUCTS_DIR = os.path.join(DATA_DIR, 'original_products')  # Database of original product images
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample_data')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ORIGINAL_PRODUCTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

# Model configurations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Siamese Network Configuration
SIAMESE_MODEL_PATH = os.path.join(MODELS_DIR, 'siamese_model.h5')
SIMILARITY_THRESHOLD = 0.85  # Products with similarity > 0.85 are considered authentic

# Damage Detection Configuration
DAMAGE_MODEL_PATH = os.path.join(MODELS_DIR, 'damage_model.h5')
DAMAGE_CLASSES = ['No Damage', 'Minor Damage', 'Major Damage', 'Severely Damaged']
NUM_DAMAGE_CLASSES = len(DAMAGE_CLASSES)

# Training configurations
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 5000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}
