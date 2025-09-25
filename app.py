"""
Flask Application for Property Image Classification
Deployed on GCP Cloud Run for scalable inference
"""

import os
import json
import base64
import io
import logging
from datetime import datetime
from typing import Dict, Any

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PropertyClassifier:
    def __init__(self, model_path: str, class_mapping_path: str):
        """Initialize the classifier with model and class mappings"""
        logger.info("Initializing PropertyClassifier...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load class mapping
        logger.info(f"Loading class mapping from: {class_mapping_path}")
        try:
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
            self.num_classes = len(self.class_mapping)
            logger.info(f"Successfully loaded {self.num_classes} classes: {list(self.class_mapping.values())}")
        except Exception as e:
            error_msg = f"Failed to load class mapping from {class_mapping_path}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        try:
            self.model = self._load_model(model_path)
            self.model.eval()
            logger.info("Model loaded and set to evaluation mode successfully")
        except Exception as e:
            error_msg = f"Failed to load model from {model_path}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Image preprocessing pipeline
        logger.info("Setting up image preprocessing pipeline...")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        logger.info("Image preprocessing pipeline initialized")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model"""
        logger.info("Creating EfficientNet-B0 model architecture...")
        
        try:
            # Create model architecture (EfficientNet-B0)
            model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, self.num_classes)
            )
            logger.info(f"Model architecture created with {self.num_classes} output classes")
            
            # Load trained weights
            logger.info(f"Loading checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if checkpoint contains expected keys
            if 'model_state_dict' not in checkpoint:
                logger.warning("Checkpoint does not contain 'model_state_dict' key, trying to load directly")
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            model.to(self.device)
            logger.info(f"Model weights loaded successfully and moved to {self.device}")
            
            return model
            
        except Exception as e:
            error_msg = f"Error loading model architecture or weights: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def preprocess_image(self, base64_string: str) -> torch.Tensor:
        """Convert base64 string to preprocessed tensor"""
        logger.info("Starting image preprocessing...")
        
        try:
            # Decode base64 to bytes
            logger.info("Decoding base64 string to bytes...")
            image_bytes = base64.b64decode(base64_string)
            logger.info(f"Successfully decoded {len(image_bytes)} bytes from base64")
            
            # Convert to PIL Image
            logger.info("Converting bytes to PIL Image...")
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Successfully opened image with size: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB...")
                image = image.convert('RGB')
                logger.info("Image converted to RGB successfully")
            
            # Apply transforms
            logger.info("Applying image transformations...")
            image_tensor = self.transform(image)  # This returns a torch.Tensor
            # Type assertion for linter
            if isinstance(image_tensor, torch.Tensor):
                logger.info(f"Image transformed to tensor with shape: {image_tensor.shape}")
                
                # Add batch dimension
                logger.info("Adding batch dimension...")
                batched_tensor = image_tensor.unsqueeze(0)
                logger.info(f"Tensor shape after adding batch dimension: {batched_tensor.shape}")
            else:
                raise ValueError("Transform did not return a tensor")
            
            # Move to device
            final_tensor = batched_tensor.to(self.device)
            logger.info(f"Tensor moved to device: {self.device}")
            
            return final_tensor
            
        except Exception as e:
            error_msg = f"Error preprocessing image: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Predict classes for multiple images with nested structure"""
        logger.info(f"Starting prediction for {len(input_data)} images...")
        results = {}
        
        with torch.no_grad():
            for image_id, image_data in input_data.items():
                logger.info(f"Processing image: {image_id}")
                
                try:
                    # Extract base64 data from nested structure
                    logger.info(f"Extracting base64 data from image_data for {image_id}...")
                    base64_data = ""
                    
                    if isinstance(image_data, dict) and "base64" in image_data:
                        logger.info(f"Found 'base64' key in image_data for {image_id}")
                        if isinstance(image_data["base64"], dict) and "data" in image_data["base64"]:
                            base64_data = str(image_data["base64"]["data"])
                            logger.info(f"Extracted nested base64 data for {image_id}")
                        else:
                            base64_data = str(image_data["base64"])
                            logger.info(f"Extracted direct base64 data for {image_id}")
                    elif isinstance(image_data, str):
                        # Fallback for direct base64 string
                        base64_data = image_data
                        logger.info(f"Using direct string as base64 data for {image_id}")
                    else:
                        error_msg = f"Invalid image data format for {image_id}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    logger.info(f"Base64 data length for {image_id}: {len(base64_data)} characters")
                    
                    # Preprocess image
                    logger.info(f"Preprocessing image {image_id}...")
                    image_tensor = self.preprocess_image(base64_data)
                    logger.info(f"Image {image_id} preprocessed successfully")
                    
                    # Get prediction
                    logger.info(f"Running model inference for {image_id}...")
                    outputs = self.model(image_tensor)
                    logger.info(f"Model inference completed for {image_id}, output shape: {outputs.shape}")
                    
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][int(predicted_class_idx)].item()
                    
                    logger.info(f"Prediction for {image_id}: class_idx={predicted_class_idx}, confidence={confidence:.4f}")
                    
                    # Map to class name
                    if str(predicted_class_idx) not in self.class_mapping:
                        error_msg = f"Predicted class index {predicted_class_idx} not found in class mapping for {image_id}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    predicted_class = self.class_mapping[str(predicted_class_idx)]
                    logger.info(f"Mapped class index {predicted_class_idx} to class name: {predicted_class} for {image_id}")
                    
                    # Store result - simple format as requested
                    results[image_id] = predicted_class
                    
                    logger.info(f"Successfully processed {image_id}: {predicted_class} (confidence: {confidence:.4f})")
                    
                except Exception as e:
                    error_msg = f"Error predicting image {image_id}: {str(e)}"
                    logger.error(error_msg)
                    results[image_id] = f"Error: {str(e)}"
        
        logger.info(f"Prediction completed for all {len(input_data)} images")
        return results

# Global classifier instance
classifier = None

def load_classifier():
    """Load the classifier with the target_95_model.pth"""
    global classifier
    
    logger.info("Starting classifier loading process...")
    
    # Model directory
    models_dir = "./models"
    logger.info(f"Checking models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        error_msg = f"Models directory not found: {models_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Models directory exists: {models_dir}")
    
    # Use specific target_95_model.pth file
    model_file = "target_95_model.pth"
    model_path = os.path.join(models_dir, model_file)
    
    logger.info(f"Looking for model file: {model_path}")
    
    if not os.path.exists(model_path):
        # List available files for debugging
        try:
            available_files = os.listdir(models_dir)
            logger.error(f"Available files in models directory: {available_files}")
        except Exception as e:
            logger.error(f"Could not list files in models directory: {str(e)}")
        
        error_msg = f"Required model file not found: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Model file found: {model_path}")
    
    # Look for class mapping file
    class_mapping_files = [f for f in os.listdir(models_dir) if f.startswith('class_mapping_') and f.endswith('.json')]
    
    if not class_mapping_files:
        error_msg = "No class mapping file found in models directory"
        logger.error(error_msg)
        try:
            available_files = os.listdir(models_dir)
            logger.error(f"Available files in models directory: {available_files}")
        except Exception as e:
            logger.error(f"Could not list files in models directory: {str(e)}")
        raise FileNotFoundError(error_msg)
    
    # Use the most recent class mapping file
    class_mapping_file = sorted(class_mapping_files)[-1]
    class_mapping_path = os.path.join(models_dir, class_mapping_file)
    
    logger.info(f"Using class mapping file: {class_mapping_path}")
    
    if not os.path.exists(class_mapping_path):
        error_msg = f"Class mapping file not found: {class_mapping_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Class mapping file found: {class_mapping_path}")
    
    try:
        logger.info("Initializing PropertyClassifier...")
        classifier = PropertyClassifier(model_path, class_mapping_path)
        logger.info("PropertyClassifier initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize PropertyClassifier: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for GCP Cloud Run"""
    logger.info("Health check endpoint called")
    
    try:
        is_model_loaded = classifier is not None
        logger.info(f"Model loaded status: {is_model_loaded}")
        
        response_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": is_model_loaded
        }
        
        logger.info(f"Health check response: {response_data}")
        return jsonify(response_data), 200
        
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    logger.info("Prediction endpoint called")
    
    try:
        if classifier is None:
            error_msg = "Model not loaded"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
        
        logger.info("Getting JSON data from request...")
        # Get JSON data
        data = request.get_json()
        
        if not data:
            error_msg = "No JSON data provided"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        logger.info(f"Received data type: {type(data)}")
        
        # Validate input format
        if not isinstance(data, dict):
            error_msg = "Input must be a JSON object"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        if not data:
            error_msg = "Empty input data"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        logger.info(f"Received prediction request for {len(data)} images: {list(data.keys())}")
        
        # Perform predictions
        logger.info("Starting prediction process...")
        results = classifier.predict(data)
        logger.info(f"Prediction completed with results: {results}")
        
        # Return results in simple format as requested
        logger.info("Returning prediction results")
        return jsonify(results), 200
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    logger.info("Model info endpoint called")
    
    try:
        if classifier is None:
            error_msg = "Model not loaded"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
        
        logger.info("Gathering model information...")
        
        info_data = {
            "model_info": {
                "num_classes": classifier.num_classes,
                "classes": list(classifier.class_mapping.values()),
                "device": str(classifier.device),
                "model_architecture": "EfficientNet-B0"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Model info: {info_data}")
        return jsonify(info_data), 200
        
    except Exception as e:
        error_msg = f"Info error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health", "/predict", "/info"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "Please check the logs for more details"
    }), 500

if __name__ == '__main__':
    # Local run (python app.py)
    try:
        logger.info("Starting Flask application...")
        logger.info("Loading classifier on startup (local mode)...")
        load_classifier()
        logger.info("Classifier loaded successfully (local mode)")

        # Get port from environment (Cloud Run sets PORT env var)
        port = int(os.environ.get('PORT', 8080))
        logger.info(f"Starting server on port: {port}")

        # Run the app
        app.run(host='0.0.0.0', port=port, debug=False)

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
else:
    # Gunicorn / Cloud Run mode
    try:
        logger.info("Loading classifier at module import (Gunicorn mode)...")
        load_classifier()
        logger.info("Classifier loaded successfully (Gunicorn mode)")
    except Exception as e:
        logger.error(f"Failed to load classifier during startup (Gunicorn): {e}")
