import os
from flask import Flask, render_template, request, redirect, url_for, flash
import utils

# Initialize Flask app
app = Flask(_name_)
app.secret_key = "deepfake_detection_app"  # Needed for flashing messages

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model at startup - using absolute path for better reliability
MODEL_PATH = os.path.abspath(os.path.join('model', 'deepfake_detection_app/model/DeepFakeDetection.h5'))

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and detection"""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and utils.allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Perform deepfake detection
        result, confidence = utils.detect_deepfake(filename)
        
        # Check for errors
        if result.startswith("ERROR"):
            flash(result)
            return redirect(url_for('index'))
        
        # Render result page
        return render_template('result.html', 
                              filename=file.filename,
                              result=result, 
                              confidence=confidence*100)
    else:
        flash('File type not allowed. Please upload a JPG, JPEG, or PNG image.')
        return redirect(url_for('index'))

# Add a test route to check if routing works
@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        return "POST request received!"
    return "GET request received!"

if _name_ == '_main_':
    # Make sure uploads directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Make sure model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check if model file exists
    if not os.path.isfile(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Please make sure your model file exists at this location.")
    
    # Load the model
    if not utils.load_deepfake_model(MODEL_PATH):
        print(f"WARNING: Could not load model from {MODEL_PATH}")
        print("The application will start, but prediction functionality will be unavailable.")
        print("Please make sure the model file exists at the specified path.")
    
    # Start the Flask application
    app.run(debug=True)
