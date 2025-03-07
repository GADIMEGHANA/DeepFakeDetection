import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback
import sys

# Global variables
IMG_SIZE = 224
model = None

def load_deepfake_model(model_path):
    """Load the trained DeepFake detection model with comprehensive debugging"""
    global model
    try:
        print("\n" + "="*50)
        print(f"DEBUGGING MODEL LOADING")
        print("="*50)
        
        # File checks
        print(f"Attempting to load model from: {model_path}")
        print(f"Absolute path: {os.path.abspath(model_path)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print(f"File size: {os.path.getsize(model_path)} bytes")
            print(f"File permissions: {oct(os.stat(model_path).st_mode)[-3:]}")
        else:
            print(f"ERROR: Model file does not exist at: {model_path}")
            print("Please check the file path and ensure the model is in the correct location.")
            return False
        
        # Environment info
        print("\nEnvironment Information:")
        print(f"Python version: {sys.version}")
        print(f"TensorFlow version: {tf._version_}")
        print(f"NumPy version: {np._version_}")
        print(f"OpenCV version: {cv2._version_}")
        
        # Try different loading approaches
        print("\nAttempting to load model...")
        
        # Approach 1: Standard loading
        try:
            print("Method 1: Standard loading")
            model = load_model(model_path)
            print(" SUCCESS: Model loaded with standard method!")
            return True
        except Exception as e1:
            print(f" Method 1 failed: {str(e1)}")
            
            # Approach 2: Loading with compile=False
            try:
                print("\nMethod 2: Loading with compile=False")
                model = load_model(model_path, compile=False)
                print(" SUCCESS: Model loaded with compile=False!")
                return True
            except Exception as e2:
                print(f" Method 2 failed: {str(e2)}")
                
                # Approach 3: Loading with custom_objects (empty dict)
                try:
                    print("\nMethod 3: Loading with empty custom_objects")
                    model = load_model(model_path, custom_objects={}, compile=False)
                    print(" SUCCESS: Model loaded with custom_objects!")
                    return True
                except Exception as e3:
                    print(f" Method 3 failed: {str(e3)}")
                    
                    # All methods failed
                    print("\n ALL LOADING METHODS FAILED")
                    print("Detailed error information:")
                    traceback.print_exc()
                    return False
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {str(e)}")
        traceback.print_exc()
        return False

def detect_deepfake(image_path):
    """Detect if an image is real or fake using the loaded model"""
    global model
    
    if model is None:
        return "ERROR: Model not loaded", 0
    
    try:
        # Read and preprocess the image
        print(f"Reading image from: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Cannot read image from {image_path}")
            return "ERROR: Cannot read image", 0
            
        print(f"Image shape: {img.shape}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img)[0][0]
        result = "FAKE" if prediction > 0.5 else "REAL"
        confidence = prediction if result == "FAKE" else 1 - prediction
        
        print(f"Prediction result: {result} with confidence {confidence:.4f}")
        return result, float(confidence)
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        traceback.print_exc()
        return f"ERROR: {str(e)}", 0

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
