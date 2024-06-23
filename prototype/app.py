from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and other required resources
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

# Define the classes
classes = ['HEALTHY', 'AIR LEAK', 'DIRTY OIL COOLER', 'HIGH ATMOSPHERE TEMP']

# SIFT Feature Extraction
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Extract BoVW features
def extract_bovw_features(descriptors, kmeans):
    num_clusters = kmeans.n_clusters
    bovw_features = np.zeros((1, num_clusters), dtype=np.float32)
    if descriptors is not None:
        words = kmeans.predict(descriptors)
        for w in words:
            bovw_features[0, w] += 1
    return bovw_features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        descriptors = extract_sift_features(image)
        bovw_features = extract_bovw_features(descriptors, kmeans)
        bovw_features = scaler.transform(bovw_features)

        prediction = model.predict(bovw_features)
        class_name = classes[prediction[0]]

        return render_template('result.html', class_name=class_name, image_url=filepath)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
