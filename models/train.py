"""
Training script for Product Fraud Detection Models
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import config
from models.model import create_siamese_model, create_damage_model, unfreeze_base_model
from models.preprocessing import load_and_preprocess_image


def generate_synthetic_training_data():
    """
    Generate synthetic training data for demonstration
    In production, replace this with real product images
    
    Returns:
        Tuple of (siamese_data, damage_data)
    """
    print("Generating synthetic training data...")
    
    # Generate random images for demonstration
    num_samples = 1000
    img_shape = (*config.IMAGE_SIZE, 3)
    
    # Siamese network data (product pairs)
    # Generate pairs of similar and dissimilar images
    images_a = []
    images_b = []
    labels = []
    
    for i in range(num_samples):
        # Create base image
        base_img = np.random.rand(*img_shape).astype(np.float32)
        
        if i % 2 == 0:  # Same product (50% of data)
            # Add small variations to simulate same product from different angles
            noise = np.random.normal(0, 0.1, img_shape).astype(np.float32)
            similar_img = np.clip(base_img + noise, 0, 1)
            images_a.append(base_img)
            images_b.append(similar_img)
            labels.append(1)  # Same product
        else:  # Different product
            # Generate completely different image
            different_img = np.random.rand(*img_shape).astype(np.float32)
            images_a.append(base_img)
            images_b.append(different_img)
            labels.append(0)  # Different product
    
    siamese_data = {
        'images_a': np.array(images_a),
        'images_b': np.array(images_b),
        'labels': np.array(labels)
    }
    
    # Damage detection data
    damage_images = []
    damage_labels = []
    
    for i in range(num_samples):
        img = np.random.rand(*img_shape).astype(np.float32)
        
        # Simulate damage by adding artifacts
        damage_level = i % config.NUM_DAMAGE_CLASSES
        
        if damage_level == 0:  # No damage
            pass  # Keep original
        elif damage_level == 1:  # Minor damage
            # Add small dark spots
            mask = np.random.rand(*config.IMAGE_SIZE) > 0.95
            img[mask] = img[mask] * 0.5
        elif damage_level == 2:  # Major damage
            # Add larger dark regions
            mask = np.random.rand(*config.IMAGE_SIZE) > 0.85
            img[mask] = img[mask] * 0.3
        else:  # Severely damaged
            # Add significant distortions
            mask = np.random.rand(*config.IMAGE_SIZE) > 0.7
            img[mask] = np.random.rand(np.sum(mask), 3) * 0.2
        
        damage_images.append(img)
        damage_labels.append(damage_level)
    
    # Convert to categorical
    damage_labels_categorical = keras.utils.to_categorical(
        damage_labels, 
        num_classes=config.NUM_DAMAGE_CLASSES
    )
    
    damage_data = {
        'images': np.array(damage_images),
        'labels': damage_labels_categorical
    }
    
    print(f"Generated {num_samples} samples for each model")
    return siamese_data, damage_data


def plot_training_history(history, model_name, save_path):
    """
    Plot and save training history
    
    Args:
        history: Keras training history object
        model_name: Name of the model
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def train_siamese_model(siamese_data, epochs=config.EPOCHS, test_mode=False):
    """
    Train the Siamese network for product authentication
    
    Args:
        siamese_data: Dictionary with images_a, images_b, and labels
        epochs: Number of training epochs
        test_mode: If True, train for fewer epochs for testing
        
    Returns:
        Trained model and history
    """
    print("\n" + "="*80)
    print("Training Siamese Network for Product Authentication")
    print("="*80 + "\n")
    
    if test_mode:
        epochs = 2
    
    # Split data
    indices = np.arange(len(siamese_data['labels']))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=config.VALIDATION_SPLIT, 
        random_state=42
    )
    
    train_a = siamese_data['images_a'][train_idx]
    train_b = siamese_data['images_b'][train_idx]
    train_labels = siamese_data['labels'][train_idx]
    
    val_a = siamese_data['images_a'][val_idx]
    val_b = siamese_data['images_b'][val_idx]
    val_labels = siamese_data['labels'][val_idx]
    
    # Create model
    model = create_siamese_model()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            config.SIAMESE_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        [train_a, train_b],
        train_labels,
        validation_data=([val_a, val_b], val_labels),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_path = os.path.join(config.MODELS_DIR, 'siamese_training_history.png')
    plot_training_history(history, 'Siamese Network', plot_path)
    
    print(f"\nSiamese model saved to {config.SIAMESE_MODEL_PATH}")
    
    return model, history


def train_damage_model(damage_data, epochs=config.EPOCHS, test_mode=False):
    """
    Train the damage detection model
    
    Args:
        damage_data: Dictionary with images and labels
        epochs: Number of training epochs
        test_mode: If True, train for fewer epochs for testing
        
    Returns:
        Trained model and history
    """
    print("\n" + "="*80)
    print("Training Damage Detection Model")
    print("="*80 + "\n")
    
    if test_mode:
        epochs = 2
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        damage_data['images'],
        damage_data['labels'],
        test_size=config.VALIDATION_SPLIT,
        random_state=42
    )
    
    # Create model
    model = create_damage_model()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            config.DAMAGE_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("Phase 1: Training with frozen base model...")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=min(epochs // 2, 20),
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning phase
    if not test_mode and epochs > 20:
        print("\nPhase 2: Fine-tuning with unfrozen layers...")
        model = unfreeze_base_model(model, num_layers_to_unfreeze=30)
        
        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs // 2,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in history1.history:
            history1.history[key].extend(history2.history[key])
    
    # Plot training history
    plot_path = os.path.join(config.MODELS_DIR, 'damage_training_history.png')
    plot_training_history(history1, 'Damage Detection', plot_path)
    
    print(f"\nDamage model saved to {config.DAMAGE_MODEL_PATH}")
    
    return model, history1


def main(test_mode=False):
    """
    Main training function
    
    Args:
        test_mode: If True, run in test mode with fewer epochs
    """
    print("Product Fraud Detection System - Model Training")
    print("="*80)
    
    # Generate training data
    siamese_data, damage_data = generate_synthetic_training_data()
    
    # Train Siamese model
    siamese_model, siamese_history = train_siamese_model(siamese_data, test_mode=test_mode)
    
    # Train damage detection model
    damage_model, damage_history = train_damage_model(damage_data, test_mode=test_mode)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nModels saved in: {config.MODELS_DIR}")
    print(f"- Siamese Model: {config.SIAMESE_MODEL_PATH}")
    print(f"- Damage Model: {config.DAMAGE_MODEL_PATH}")
    print("\nYou can now run the API server with: python api/app.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Product Fraud Detection Models')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with fewer epochs')
    
    args = parser.parse_args()
    
    if args.test_mode:
        print("Running in TEST MODE (2 epochs only)")
    
    main(test_mode=args.test_mode)
