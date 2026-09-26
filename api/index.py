from flask import Flask, request, jsonify
import joblib
import re
import os

app = Flask(__name__)

# Resolve path to the pickle files in the root directory
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, '..'))

vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')
model_path = os.path.join(root_dir, 'gbc_model.pkl')

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_text = data.get('text', '')

        if not raw_text.strip():
            return jsonify({'error': 'Please provide text to evaluate.'}), 400

        cleaned = clean_text(raw_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        result = "Not A Fake News" if prediction == 1 else "Fake News"
        return jsonify({'prediction': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
