from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Ensure you use the correct vectorizer used for training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = [data['text']]  # Wrap in a list for vectorization
    transformed_text = vectorizer.transform(message)  # Convert text to numerical form

    # Make prediction
    prediction = model.predict(transformed_text)
    output = "Spam" if prediction[0] == 1 else "Not Spam"

    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
