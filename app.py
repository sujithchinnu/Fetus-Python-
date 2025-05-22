from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained model, scaler, and healthy ranges
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
healthy_ranges = joblib.load("healthy_ranges.pkl")

# Initialize the Flask application
app = Flask(__name__)

# Route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the form data
        heart_rate = float(request.form['heart_rate'])
        movement = float(request.form['movement'])
        oxygen_level = float(request.form['oxygen_level'])
        temperature = float(request.form['temperature'])
        blood_pressure = float(request.form['blood_pressure'])
        respiration_rate = float(request.form['respiration_rate'])

        # Create a dictionary of the input values
        input_values = {
            'Heart Rate': heart_rate,
            'Movement': movement,
            'Oxygen Level': oxygen_level,
            'Temperature': temperature,
            'Blood Pressure': blood_pressure,
            'Respiration Rate': respiration_rate
        }

        # Check if any input is out of the healthy ranges
        out_of_range = False
        for feature, value in input_values.items():
            if not (healthy_ranges[feature]['min'] <= value <= healthy_ranges[feature]['max']):
                out_of_range = True
                break

        if out_of_range:
            # If any value is out of range, display the "Unhealthy" message
            result = "Unhealthy. Please consult a doctor."
        else:
            # Otherwise, proceed with scaling and prediction
            input_data = pd.DataFrame([input_values])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            result = 'Healthy' if prediction == 1 else 'Unhealthy. Please consult a doctor.'

        # Render the result page with the result
        return render_template('result.html', result=result)

    except Exception as e:
        # Handle errors and show an error page
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
