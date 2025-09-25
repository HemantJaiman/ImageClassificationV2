#!/usr/bin/env python3
"""
Test script for Property Classification Flask API
Tests both local development and deployed Cloud Run service
"""

import base64
import json
import requests
import time
from pathlib import Path
from typing import Dict, Any

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def test_health_endpoint(base_url: str) -> bool:
    """Test the health check endpoint"""
    try:
        print(f"\n🔍 Testing health endpoint: {base_url}/health")
        response = requests.get(f"{base_url}/health", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_info_endpoint(base_url: str) -> Dict[str, Any]:
    """Test the model info endpoint"""
    try:
        print(f"\n📊 Testing info endpoint: {base_url}/info")
        response = requests.get(f"{base_url}/info", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            model_info = data.get('model_info', {})
            print(f"✅ Model info retrieved")
            print(f"   Architecture: {model_info.get('model_architecture')}")
            print(f"   Number of classes: {model_info.get('num_classes')}")
            print(f"   Device: {model_info.get('device')}")
            print(f"   Classes: {model_info.get('classes', [])[:5]}...")  # Show first 5
            return data
        else:
            print(f"❌ Info endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {}
            
    except Exception as e:
        print(f"❌ Info endpoint error: {e}")
        return {}

def test_prediction_endpoint(base_url: str, test_images: Dict[str, str]) -> Dict[str, Any]:
    """Test the prediction endpoint with sample images"""
    try:
        print(f"\n🎯 Testing prediction endpoint: {base_url}/predict")
        print(f"   Sending {len(test_images)} test images...")
        
        headers = {'Content-Type': 'application/json'}
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict", 
            json=test_images, 
            headers=headers, 
            timeout=120
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful")
            print(f"   Processing time: {end_time - start_time:.2f} seconds")
            print(f"   Total images: {data.get('total_images')}")
            
            predictions = data.get('predictions', {})
            print(f"\n📋 Prediction Results:")
            for image_id, result in predictions.items():
                if 'error' in result:
                    print(f"   Image {image_id}: ❌ {result['error']}")
                else:
                    predicted_class = result.get('predicted_class', 'Unknown')
                    confidence = result.get('confidence_percentage', 'N/A')
                    print(f"   Image {image_id}: {predicted_class} ({confidence})")
            
            return data
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {}
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {}

def create_sample_test_data() -> Dict[str, str]:
    """Create sample base64 test data"""
    # Create a simple test image (1x1 pixel) for testing
    from PIL import Image
    import io
    
    test_images = {}
    
    # Create small test images
    for i in range(1, 4):
        # Create a small colored image
        img = Image.new('RGB', (100, 100), color=(i*80, i*60, i*40))
        
        # Convert to base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        test_images[str(i)] = img_base64
    
    return test_images

def test_api_comprehensive(base_url: str, use_sample_data: bool = True):
    """Comprehensive API testing"""
    print(f"🚀 Starting comprehensive API test for: {base_url}")
    print("=" * 60)
    
    # Test 1: Health check
    health_ok = test_health_endpoint(base_url)
    if not health_ok:
        print("❌ Stopping tests - health check failed")
        return
    
    # Test 2: Model info
    info_data = test_info_endpoint(base_url)
    if not info_data:
        print("⚠️ Warning: Could not retrieve model info")
    
    # Test 3: Prediction
    if use_sample_data:
        print("\n📝 Using generated sample images for testing")
        test_images = create_sample_test_data()
    else:
        # Look for actual image files in current directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path('.').glob(f'*{ext}'))
            image_files.extend(Path('.').glob(f'*{ext.upper()}'))
        
        if image_files:
            print(f"\n📸 Found {len(image_files)} image files, using first 3 for testing")
            test_images = {}
            for i, img_file in enumerate(image_files[:3], 1):
                base64_data = encode_image_to_base64(str(img_file))
                if base64_data:
                    test_images[str(i)] = base64_data
                    print(f"   Loaded: {img_file}")
        else:
            print("\n📝 No image files found, using generated sample images")
            test_images = create_sample_test_data()
    
    if test_images:
        prediction_data = test_prediction_endpoint(base_url, test_images)
    else:
        print("❌ No test images available")
    
    print(f"\n{'='*60}")
    print("🏁 API Testing Complete!")
    print(f"{'='*60}")

def main():
    """Main test function"""
    print("🧪 Property Classification API Test Suite")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {
            "name": "Local Development Server",
            "url": "http://localhost:8080",
            "description": "Test local Flask development server"
        },
        {
            "name": "Local Docker Container",
            "url": "http://localhost:8080",
            "description": "Test local Docker container"
        }
    ]
    
    # Add Cloud Run URL if available
    cloud_run_url = input("\n🌐 Enter your Cloud Run service URL (or press Enter to skip): ").strip()
    if cloud_run_url:
        if not cloud_run_url.startswith('http'):
            cloud_run_url = f"https://{cloud_run_url}"
        
        test_configs.append({
            "name": "GCP Cloud Run Service",
            "url": cloud_run_url,
            "description": "Test deployed Cloud Run service"
        })
    
    # Run tests for each configuration
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*20} Test {i}: {config['name']} {'='*20}")
        print(f"Description: {config['description']}")
        print(f"URL: {config['url']}")
        
        try:
            test_api_comprehensive(config['url'])
        except KeyboardInterrupt:
            print("\n⏹️ Testing interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
        
        if i < len(test_configs):
            input("\nPress Enter to continue to next test...")

if __name__ == "__main__":
    main()