# Property Classification API Usage Guide

## Updated for entry_sample.json Format

The Flask API has been updated to handle the nested JSON structure found in your [entry_sample.json]file.

## Input Format

**Expected JSON Structure:**
```json
{
  "1": {
    "base64": {
      "data": "base64_encoded_image_data_here"
    }
  },
  "2": {
    "base64": {
      "data": "base64_encoded_image_data_here"
    }
  },
  "3": {
    "base64": {
      "data": "base64_encoded_image_data_here"
    }
  }
}
```

## Output Format

**Simple Response (as requested):**
```json
{
  "1": "predicted_class_name",
  "2": "predicted_class_name", 
  "3": "predicted_class_name"
}
```

**Example Response:**
```json
{
  "1": "apartment",
  "2": "house",
  "3": "office"
}
```

## API Endpoints

### 1. Health Check
```http
GET /health
```

### 2. Model Information
```http
GET /info
```

### 3. Prediction (Main Endpoint)
```http
POST /predict
Content-Type: application/json
```

## Error Handling

If an image fails to process, the response will include an error message:
```json
{
  "1": "apartment",
  "2": "Error: Invalid image data",
  "3": "house"
}
```

## Key Features

1. **Automatic Model Selection**: Uses `target_95_model_*.pth` if available (models with 95%+ accuracy), otherwise falls back to `best_model_*.pth`

2. **Flexible Input Support**: 
   - Supports nested structure from entry_sample.json
   - Also supports direct base64 strings as fallback

3. **Simple Output**: Returns just the predicted class names as requested

4. **Batch Processing**: Can handle multiple images in a single request

5. **Error Resilience**: Individual image failures don't stop processing of other images

## Testing

Use the provided `test_api.py` script:

```bash
python test_api.py
```

This will test:
- Health endpoint
- Model info endpoint  
- Prediction endpoint with sample data

## Deployment

Follow the instructions in [DEPLOYMENT_GUIDE.md] to deploy to GCP Cloud Run.

## Model Requirements

Ensure you have:
- A trained model file (`target_95_model_*.pth` or `best_model_*.pth`) in the `./models` folder
- Corresponding class mapping file (`class_mapping_*.json`) in the same folder

The API will automatically load the best available model when starting up.