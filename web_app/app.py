from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib

matplotlib.use('Agg')  # Required to generate images without a screen
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join('..', 'models', 'best_model.pkl')
FEATURES_PATH = os.path.join('..', 'models', 'feature_names.pkl')

# Load Artifacts
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)


def generate_radar_chart(input_data):
    """
    Generates a Radar Chart to visualize where the machine parameters
    stand compared to standard operating limits.
    Returns: Base64 encoded image string.
    """
    # Normalize values for visualization (simplification for display)
    # We define arbitrary "max safe values" based on dataset knowledge for the visual
    categories = ['Rot_Speed', 'Torque', 'Tool_Wear', 'Heat_Dissipation']

    # Extract values
    speed_score = min(input_data['Rot_Speed'] / 2800, 1.0)  # Max speed approx 2886
    torque_score = min(input_data['Torque'] / 70, 1.0)  # Max torque approx 76
    wear_score = min(input_data['Tool_Wear'] / 250, 1.0)  # Failures often > 200
    heat_score = min((input_data['Process_Temp'] - input_data['Air_Temp']) / 12, 1.0)

    values = [speed_score, torque_score, wear_score, heat_score]

    # Close the loop for the radar chart
    values += [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#1abc9c', alpha=0.25)
    ax.plot(angles, values, color='#1abc9c', linewidth=2)

    # Fix the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Add a "Danger Zone" ring at 0.8
    ax.plot(np.linspace(0, 2 * np.pi, 100), [0.8] * 100, color='red', linestyle='--', linewidth=1,
            label='Danger Threshold')

    plt.title('Component Stress Analysis', size=15, color='#34495e', y=1.1)

    # Convert to Base64 string to embed in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get Raw Input from Form
        air_temp = float(request.form['air_temp'])
        process_temp = float(request.form['process_temp'])
        rot_speed = float(request.form['rot_speed'])
        torque = float(request.form['torque'])
        tool_wear = float(request.form['tool_wear'])
        type_val = request.form['type']  # 'L', 'M', or 'H'

        # 2. Physics-Based Feature Engineering (Must match training!)
        temp_diff = process_temp - air_temp
        power_factor = rot_speed * torque
        tool_strain = tool_wear * torque

        # 3. Create DataFrame with correct columns
        input_data = {
            'Type': [type_val],
            'Air_Temp': [air_temp],
            'Process_Temp': [process_temp],
            'Rot_Speed': [rot_speed],
            'Torque': [torque],
            'Tool_Wear': [tool_wear],
            'Temp_Diff': [temp_diff],
            'Power_Factor': [power_factor],
            'Tool_Strain': [tool_strain]
        }

        df = pd.DataFrame(input_data)

        # Ensure column order matches training
        # (We use the feature names saved during training, excluding ones we just engineered if names differ,
        # but here we reconstructed exactly.)

        # 4. Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of Class 1 (Failure)

        # 5. Generate Visuals
        radar_chart = generate_radar_chart(df.iloc[0])

        # 6. Interpret Results
        risk_level = "LOW"
        color = "green"
        if probability > 0.4:
            risk_level = "MODERATE"
            color = "orange"
        if probability > 0.75:
            risk_level = "CRITICAL"
            color = "red"

        return render_template('result.html',
                               prob=round(probability * 100, 2),
                               risk=risk_level,
                               color=color,
                               chart=radar_chart,
                               details=input_data)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True, port=5000)