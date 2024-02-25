from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the trained model
model = joblib.load(open(r'C:\Users\Sakshi\OneDrive\Desktop\wine\FINAL_Wine_Quality_Model.joblib', 'rb'))

# Mapping function for prediction labels
def map_prediction_label(prediction):
    if prediction == 0:
        return "Low Quality"
    elif prediction == 1:
        return "High Quality"
    else:
        return "Unknown"

# Function to map category labels to numerical values
def map_category_to_value(category, category_mapping):
    return category_mapping.categories.get_loc(category)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Mapping of category labels to numerical values
    category_mappings = {
        'fixed_acidity_category': pd.CategoricalDtype(categories=['0-4', '4-7', '7-9', '9-12', '12-16']),
        'volatile_acidity_category': pd.CategoricalDtype(categories=['0-0.5', '0.5-1', '1-1.5']),
        'chlorides_category': pd.CategoricalDtype(categories=['0-0.08', '0.08-0.1', '0.1-0.2', '0.2-0.611']),
        'total_sulfur_dioxide_category': pd.CategoricalDtype(categories=['0-50', '50-100', '100-150', '150-200', '200-289']),
        'pH_category': pd.CategoricalDtype(categories=['2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.1']),
        'sulphates_category': pd.CategoricalDtype(categories=['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0']),
        'alcohol_category': pd.CategoricalDtype(categories=['8.0-10.0', '10.0-12.0', '12.0-14.0', '14.0-15.0'])
    }

    # Get values from the form
    input_data = []
    for feature in category_mappings.keys():
        category = request.form[feature]
        numerical_value = map_category_to_value(category, category_mappings[feature])
        input_data.append(numerical_value)

    try:
        # Create input data as a numpy array
        input_data = np.array(input_data).reshape(1, -1)

        # Make prediction
        raw_prediction = model.predict(input_data)
        prediction_label = map_prediction_label(raw_prediction[0])
    except Exception as e:
        print("Prediction Error:", str(e))
        prediction_label = "Error: Unable to make a prediction."

    return render_template('index.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True