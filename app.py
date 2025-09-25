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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load class mapping
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
        self.num_classes = len(self.class_mapping)
        logger.info(f"Loaded {self.num_classes} classes")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model"""
        # Create model architecture (EfficientNet-B0)
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, self.num_classes)
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, base64_string: str) -> torch.Tensor:
        """Convert base64 string to preprocessed tensor"""
        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Predict classes for multiple images with nested structure"""
        results = {}
        
        with torch.no_grad():
            for image_id, image_data in input_data.items():
                try:
                    # Extract base64 data from nested structure
                    base64_data = ""
                    if isinstance(image_data, dict) and "base64" in image_data:
                        if isinstance(image_data["base64"], dict) and "data" in image_data["base64"]:
                            base64_data = str(image_data["base64"]["data"])
                        else:
                            base64_data = str(image_data["base64"])
                    elif isinstance(image_data, str):
                        # Fallback for direct base64 string
                        base64_data = image_data
                    else:
                        raise ValueError("Invalid image data format")
                    
                    # Preprocess image
                    image_tensor = self.preprocess_image(base64_data)
                    
                    # Get prediction
                    outputs = self.model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][int(predicted_class_idx)].item()
                    
                    # Map to class name
                    predicted_class = self.class_mapping[str(predicted_class_idx)]
                    
                    # Store result - simple format as requested
                    results[image_id] = predicted_class
                    
                    logger.info(f"Image {image_id}: {predicted_class} ({confidence:.4f})")
                    
                except Exception as e:
                    logger.error(f"Error predicting image {image_id}: {str(e)}")
                    results[image_id] = f"Error: {str(e)}"
        
        return results

# Global classifier instance
classifier = None

def load_classifier():
    """Load the classifier with the best available model"""
    global classifier
    
    # Model directory
    models_dir = "./models"
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find the best model (prioritize target_95_model, then best_model)
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    target_models = [f for f in model_files if f.startswith('target_95_model')]
    best_models = [f for f in model_files if f.startswith('best_model')]
    
    if target_models:
        # Use the most recent target model
        model_file = sorted(target_models)[-1]
        logger.info(f"Using target model: {model_file}")
    elif best_models:
        # Use the most recent best model
        model_file = sorted(best_models)[-1]
        logger.info(f"Using best model: {model_file}")
    else:
        raise FileNotFoundError("No trained models found in ./models directory")
    
    model_path = os.path.join(models_dir, model_file)
    
    # Find corresponding class mapping
    timestamp = model_file.split('_')[-1].replace('.pth', '')
    class_mapping_file = f"class_mapping_{timestamp}.json"
    class_mapping_path = os.path.join(models_dir, class_mapping_file)
    
    if not os.path.exists(class_mapping_path):
        # Try to find any class mapping file
        mapping_files = [f for f in os.listdir(models_dir) if f.startswith('class_mapping_')]
        if mapping_files:
            class_mapping_file = sorted(mapping_files)[-1]
            class_mapping_path = os.path.join(models_dir, class_mapping_file)
            logger.warning(f"Using alternative class mapping: {class_mapping_file}")
        else:
            raise FileNotFoundError("No class mapping file found")
    
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Loading class mapping: {class_mapping_path}")
    
    classifier = PropertyClassifier(model_path, class_mapping_path)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for GCP Cloud Run"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": classifier is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if classifier is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input format
        if not isinstance(data, dict):
            return jsonify({"error": "Input must be a JSON object"}), 400
        
        if not data:
            return jsonify({"error": "Empty input data"}), 400
        
        logger.info(f"Received prediction request for {len(data)} images")
        
        # Perform predictions
        results = classifier.predict(data)
        
        # Return results in simple format as requested
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if classifier is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        return jsonify({
            "model_info": {
                "num_classes": classifier.num_classes,
                "classes": list(classifier.class_mapping.values()),
                "device": str(classifier.device),
                "model_architecture": "EfficientNet-B0"
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Info error: {str(e)}")
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
    try:
        # Load the classifier on startup
        load_classifier()
        
        # Get port from environment (Cloud Run sets PORT env var)
        port = int(os.environ.get('PORT', 8080))
        
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise