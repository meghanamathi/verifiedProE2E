"""
Utility functions for the API
"""
import os
import numpy as np
from PIL import Image
import io
from werkzeug.utils import secure_filename
import config


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension
    
    Args:
        filename: Name of the file
        
    Returns:
        Boolean indicating if file is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def process_uploaded_file(file):
    """
    Process an uploaded file and convert to PIL Image
    
    Args:
        file: FileStorage object from Flask request
        
    Returns:
        PIL Image object
    """
    # Read file data
    file_data = file.read()
    
    # Convert to PIL Image
    img = Image.open(io.BytesIO(file_data))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def get_original_product_image(product_id):
    """
    Retrieve original product image from database/storage
    In production, this would query your database
    
    Args:
        product_id: Product identifier
        
    Returns:
        PIL Image object or None if not found
    """
    # Construct path to original product image
    image_path = os.path.join(config.ORIGINAL_PRODUCTS_DIR, f"{product_id}.png")
    
    # Check if image exists
    if not os.path.exists(image_path):
        # Try other extensions
        for ext in ['jpg', 'jpeg', 'webp']:
            alt_path = os.path.join(config.ORIGINAL_PRODUCTS_DIR, f"{product_id}.{ext}")
            if os.path.exists(alt_path):
                image_path = alt_path
                break
        else:
            return None
    
    # Load and return image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def format_verification_response(similarity_score, damage_class, damage_confidence):
    """
    Format the API response for product verification
    
    Args:
        similarity_score: Similarity score from Siamese network (0-1)
        damage_class: Predicted damage class
        damage_confidence: Confidence score for damage prediction
        
    Returns:
        Dictionary with formatted response
    """
    is_authentic = similarity_score >= config.SIMILARITY_THRESHOLD
    
    # Determine overall status
    if not is_authentic:
        status = "REJECTED - Product Mismatch"
        recommendation = "The returned product does not match the original. Return rejected."
        color = "red"
    elif damage_class == "No Damage":
        status = "APPROVED - Full Refund"
        recommendation = "Product is authentic and undamaged. Approve full refund."
        color = "green"
    elif damage_class == "Minor Damage":
        status = "APPROVED - Partial Refund"
        recommendation = "Product is authentic with minor damage. Consider partial refund."
        color = "orange"
    else:  # Major or Severe damage
        status = "REJECTED - Excessive Damage"
        recommendation = "Product has significant damage beyond acceptable return condition."
        color = "red"
    
    return {
        'status': status,
        'recommendation': recommendation,
        'color': color,
        'authentication': {
            'is_authentic': is_authentic,
            'similarity_score': float(similarity_score),
            'threshold': config.SIMILARITY_THRESHOLD,
            'confidence': 'High' if abs(similarity_score - config.SIMILARITY_THRESHOLD) > 0.15 else 'Medium'
        },
        'damage_assessment': {
            'damage_level': damage_class,
            'confidence': float(damage_confidence),
            'all_predictions': {}  # Will be filled by caller
        }
    }


def create_error_response(message, status_code=400):
    """
    Create a standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    return {
        'error': True,
        'message': message,
        'status_code': status_code
    }, status_code
