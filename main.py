from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
labels = ['PNEUMONIA', 'NORMAL']
img_size = 128

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'file' not in request.files:
            return "No file part in the request.", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected for uploading.", 400
        if file:
            # Save the uploaded image
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the uploaded image
            try:
                img_arr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data = np.array(resized_arr).reshape(-1, img_size, img_size, 1) / 255.0

                # Load the pre-trained model
                model = load_model('Sequential.h5')  # Replace with your model path

                # Predict using the model
                prediction = model.predict(data)
                predicted_label = labels[int(prediction[0] > 0.5)]

                return render_template('index.html', prediction=predicted_label)
            except Exception as e:
                return f"Error processing the image: {e}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)