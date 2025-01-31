from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall']),
        ]
        
        # Predict using the model
        prediction = model.predict([data])
        return render_template('index.html', prediction_text=f"Predicted Crop: {prediction[0]}")
    except Exception as e:
        return render_template('index.html', error_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
