"""
Image preprocessing utilities for Product Fraud Detection System
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


def load_and_preprocess_image(image_path, target_size=config.IMAGE_SIZE):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the image file or PIL Image object
        target_size: Tuple of (height, width) to resize image
        
    Returns:
        Preprocessed image as numpy array
    """
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array


def preprocess_image_batch(images, target_size=config.IMAGE_SIZE):
    """
    Preprocess a batch of images
    
    Args:
        images: List of image paths or PIL Image objects
        target_size: Tuple of (height, width) to resize images
        
    Returns:
        Batch of preprocessed images as numpy array
    """
    processed_images = []
    
    for img in images:
        processed_img = load_and_preprocess_image(img, target_size)
        processed_images.append(processed_img)
    
    return np.array(processed_images)


def create_data_generator(augment=True):
    """
    Create an image data generator for training
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        ImageDataGenerator object
    """
    if augment:
        return ImageDataGenerator(
            rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=config.AUGMENTATION_CONFIG['shear_range'],
            zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
            fill_mode=config.AUGMENTATION_CONFIG['fill_mode'],
            rescale=1./255
        )
    else:
        return ImageDataGenerator(rescale=1./255)


def prepare_siamese_pairs(images1, images2, labels):
    """
    Prepare image pairs for Siamese network training
    
    Args:
        images1: First set of images
        images2: Second set of images
        labels: Binary labels (1 for same product, 0 for different)
        
    Returns:
        Tuple of ([images1, images2], labels)
    """
    return [np.array(images1), np.array(images2)], np.array(labels)


def augment_image(image):
    """
    Apply random augmentation to a single image
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Augmented image
    """
    # Random rotation
    angle = np.random.uniform(-20, 20)
    image = tf.keras.preprocessing.image.apply_affine_transform(
        image, theta=angle, fill_mode='nearest'
    )
    
    # Random brightness
    brightness_factor = np.random.uniform(0.8, 1.2)
    image = tf.image.adjust_brightness(image, brightness_factor)
    
    # Random flip
    if np.random.random() > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image.numpy()
