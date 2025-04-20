from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from ML_engine import predict_clinical, predict_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        if 'clinical_data' not in request.form:
            return jsonify({'error': 'No clinical data provided'}), 400
        image_file = request.files['image']
        clinical_data = json.loads(request.form['clinical_data'])
        clinical_result = predict_clinical(clinical_data)
        image_result = predict_image(image_file)
        irregular_period = clinical_data.get('Cycle(R/I)', 0) == 4
        clinical_positive = clinical_result['prediction'] == 'PCOS'
        ultrasound_positive = image_result['prediction'] == 'PCOS'
        positive_criteria_count = sum([
            clinical_positive,
            ultrasound_positive,
            irregular_period
        ])
        rotterdam_result = {
            "Final Prediction according to Rotterdam criteria": "PCOS POSITIVE" if positive_criteria_count >= 2 else "PCOS NEGATIVE",
            "Confidence level": 
                round(
                (clinical_result['probabilities'].get('PCOS', 0) * 100 +
                 image_result['probabilities'].get('PCOS', 0) * 100 +
                 (100 if irregular_period else 0)) / 3,
                2
            )
        }
        return jsonify({
            'clinical_prediction': clinical_result,
            'image_prediction': image_result,
            'rotterdam_criteria_result': rotterdam_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)