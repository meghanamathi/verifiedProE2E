"""
Flask API for Product Fraud Detection System
"""
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import config
from models.preprocessing import load_and_preprocess_image
from api.utils import (
    allowed_file, 
    process_uploaded_file, 
    get_original_product_image,
    format_verification_response,
    create_error_response
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE

# Load models (will be loaded on first request)
siamese_model = None
damage_model = None


def load_models():
    """
    Load the trained models
    """
    global siamese_model, damage_model
    
    if siamese_model is None:
        if os.path.exists(config.SIAMESE_MODEL_PATH):
            print(f"Loading Siamese model from {config.SIAMESE_MODEL_PATH}")
            siamese_model = keras.models.load_model(config.SIAMESE_MODEL_PATH)
            print("Siamese model loaded successfully")
        else:
            print(f"WARNING: Siamese model not found at {config.SIAMESE_MODEL_PATH}")
            print("Please train the model first using: python models/train.py")
    
    if damage_model is None:
        if os.path.exists(config.DAMAGE_MODEL_PATH):
            print(f"Loading Damage model from {config.DAMAGE_MODEL_PATH}")
            damage_model = keras.models.load_model(config.DAMAGE_MODEL_PATH)
            print("Damage model loaded successfully")
        else:
            print(f"WARNING: Damage model not found at {config.DAMAGE_MODEL_PATH}")
            print("Please train the model first using: python models/train.py")


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'siamese_model_loaded': siamese_model is not None,
        'damage_model_loaded': damage_model is not None,
        'models_path': config.MODELS_DIR
    })


@app.route('/api/verify-product', methods=['POST'])
def verify_product():
    """
    Verify if returned product matches original
    Expects: product_id and returned_image file
    """
    try:
        # Load models if not already loaded
        load_models()
        
        if siamese_model is None:
            return create_error_response(
                "Siamese model not loaded. Please train the model first.",
                500
            )
        
        # Get product ID
        if 'product_id' not in request.form:
            return create_error_response("Missing product_id parameter")
        
        product_id = request.form['product_id']
        
        # Get returned product image
        if 'returned_image' not in request.files:
            return create_error_response("Missing returned_image file")
        
        returned_file = request.files['returned_image']
        
        if returned_file.filename == '':
            return create_error_response("No file selected")
        
        if not allowed_file(returned_file.filename):
            return create_error_response(
                f"Invalid file type. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )
        
        # Process returned image
        returned_img = process_uploaded_file(returned_file)
        
        # Get original product image from database
        original_img = get_original_product_image(product_id)
        
        if original_img is None:
            return create_error_response(
                f"Original product image not found for product_id: {product_id}",
                404
            )
        
        # Preprocess images
        original_processed = load_and_preprocess_image(original_img)
        returned_processed = load_and_preprocess_image(returned_img)
        
        # Add batch dimension
        original_batch = np.expand_dims(original_processed, axis=0)
        returned_batch = np.expand_dims(returned_processed, axis=0)
        
        # Predict similarity
        similarity_score = siamese_model.predict([original_batch, returned_batch], verbose=0)[0][0]
        
        is_authentic = similarity_score >= config.SIMILARITY_THRESHOLD
        
        return jsonify({
            'product_id': product_id,
            'is_authentic': bool(is_authentic),
            'similarity_score': float(similarity_score),
            'threshold': config.SIMILARITY_THRESHOLD,
            'confidence': 'High' if abs(similarity_score - config.SIMILARITY_THRESHOLD) > 0.15 else 'Medium'
        })
    
    except Exception as e:
        return create_error_response(f"Error processing request: {str(e)}", 500)


@app.route('/api/assess-damage', methods=['POST'])
def assess_damage():
    """
    Assess damage level of a product image
    Expects: product_image file
    """
    try:
        # Load models if not already loaded
        load_models()
        
        if damage_model is None:
            return create_error_response(
                "Damage model not loaded. Please train the model first.",
                500
            )
        
        # Get product image
        if 'product_image' not in request.files:
            return create_error_response("Missing product_image file")
        
        image_file = request.files['product_image']
        
        if image_file.filename == '':
            return create_error_response("No file selected")
        
        if not allowed_file(image_file.filename):
            return create_error_response(
                f"Invalid file type. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )
        
        # Process image
        img = process_uploaded_file(image_file)
        
        # Preprocess image
        img_processed = load_and_preprocess_image(img)
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Predict damage
        predictions = damage_model.predict(img_batch, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = config.DAMAGE_CLASSES[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        # Create response with all class probabilities
        all_predictions = {
            config.DAMAGE_CLASSES[i]: float(predictions[i])
            for i in range(len(config.DAMAGE_CLASSES))
        }
        
        return jsonify({
            'damage_level': predicted_class,
            'confidence': float(confidence),
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        return create_error_response(f"Error processing request: {str(e)}", 500)


@app.route('/api/full-analysis', methods=['POST'])
def full_analysis():
    """
    Perform complete product verification and damage assessment
    Expects: product_id and returned_image file
    """
    try:
        # Load models if not already loaded
        load_models()
        
        if siamese_model is None or damage_model is None:
            return create_error_response(
                "Models not loaded. Please train the models first.",
                500
            )
        
        # Get product ID
        if 'product_id' not in request.form:
            return create_error_response("Missing product_id parameter")
        
        product_id = request.form['product_id']
        
        # Get returned product image
        if 'returned_image' not in request.files:
            return create_error_response("Missing returned_image file")
        
        returned_file = request.files['returned_image']
        
        if returned_file.filename == '':
            return create_error_response("No file selected")
        
        if not allowed_file(returned_file.filename):
            return create_error_response(
                f"Invalid file type. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )
        
        # Process returned image
        returned_img = process_uploaded_file(returned_file)
        
        # Get original product image from database
        original_img = get_original_product_image(product_id)
        
        if original_img is None:
            return create_error_response(
                f"Original product image not found for product_id: {product_id}",
                404
            )
        
        # Preprocess images
        original_processed = load_and_preprocess_image(original_img)
        returned_processed = load_and_preprocess_image(returned_img)
        
        # Add batch dimension
        original_batch = np.expand_dims(original_processed, axis=0)
        returned_batch = np.expand_dims(returned_processed, axis=0)
        
        # 1. Verify product authenticity
        similarity_score = siamese_model.predict([original_batch, returned_batch], verbose=0)[0][0]
        
        # 2. Assess damage
        damage_predictions = damage_model.predict(returned_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(damage_predictions)
        predicted_class = config.DAMAGE_CLASSES[predicted_class_idx]
        damage_confidence = damage_predictions[predicted_class_idx]
        
        # Create all predictions dict
        all_predictions = {
            config.DAMAGE_CLASSES[i]: float(damage_predictions[i])
            for i in range(len(config.DAMAGE_CLASSES))
        }
        
        # Format response
        response = format_verification_response(
            similarity_score,
            predicted_class,
            damage_confidence
        )
        
        response['damage_assessment']['all_predictions'] = all_predictions
        response['product_id'] = product_id
        
        return jsonify(response)
    
    except Exception as e:
        return create_error_response(f"Error processing request: {str(e)}", 500)


if __name__ == '__main__':
    print("="*80)
    print("Product Fraud Detection API Server")
    print("="*80)
    print(f"\nServer starting on http://{config.API_HOST}:{config.API_PORT}")
    print("\nAvailable endpoints:")
    print("  GET  /api/health          - Health check")
    print("  POST /api/verify-product  - Verify product authenticity")
    print("  POST /api/assess-damage   - Assess product damage")
    print("  POST /api/full-analysis   - Complete verification + damage assessment")
    print("\n" + "="*80 + "\n")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=True
    )
