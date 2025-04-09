from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import numpy as np
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///spam_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load your pre-trained model
model = load_model('spam_classifier_model.h5')

# Load the tokenizer from a saved JSON file
with open('tokenizer.json', 'r') as f:
    tokenizer_data = f.read()  # Read the tokenizer data as a string
    tokenizer = tokenizer_from_json(tokenizer_data)  # Load the tokenizer

# Define the maxlen (should be the same as used during training)
maxlen = 100  # Adjust as necessary

# Define a database model for predictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String, nullable=False)
    prediction_result = db.Column(db.String, nullable=False)

# Create the database if it doesn't exist
if not os.path.exists('spam_predictions.db'):
    with app.app_context():
        db.create_all()  # Create the database and tables

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = Prediction.query.all()  # Fetch all predictions from the database
    prediction_label = None  # Initialize prediction label to None
    if request.method == 'POST':
        email_content = request.form['email_content']
        # Preprocess the email (ensure it’s passed as a list)
        sequence = tokenizer.texts_to_sequences([email_content])  # Wrap email in a list
        padded = pad_sequences(sequence, maxlen=maxlen)
        
        # Make prediction
        prediction = model.predict(padded)
        
        # Get prediction probabilities
        spam_probability = prediction[0][0]  # Model outputs a probability, extract it
        prediction_label = "Spam" if spam_probability >= 0.5 else "Not Spam"
        
        # Save the prediction to the database
        new_prediction = Prediction(input_text=email_content, prediction_result=prediction_label)
        db.session.add(new_prediction)
        db.session.commit()

        # Fetch all predictions again to include the latest one
        predictions = Prediction.query.all()  # Update the predictions list

    return render_template('index.html', predictions=predictions, latest_prediction=prediction_label)

@app.route('/delete/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    prediction_to_delete = Prediction.query.get_or_404(prediction_id)
    db.session.delete(prediction_to_delete)
    db.session.commit()
    return redirect(url_for('index'))  # Redirect back to the index page

if __name__ == '__main__':
    app.run(debug=True)
