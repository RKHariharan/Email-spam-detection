from flask import Flask, render_template, request
import joblib
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
lg_model = joblib.load('lg.pkl')
feature_extraction = joblib.load('vectorizer.pkl')

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String(500))
    prediction_result = db.Column(db.String(10))

    def __repr__(self):
        return f'<Prediction {self.id}>'

with app.app_context():
    db.create_all()

def preprocess_text(text):
    return text

@app.route('/')
def index():
    predictions = Prediction.query.all()
    return render_template('index.html', predictions=predictions)

@app.route('/hariharan', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_content = request.form['email_content']
        processed_content = preprocess_text(email_content)
        input_data_features = feature_extraction.transform([processed_content])
        prediction = lg_model.predict(input_data_features)[0]
        result = "Spam" if prediction == 0 else "Not Spam"
        
        new_prediction = Prediction(input_text=email_content, prediction_result=result)
        db.session.add(new_prediction)
        db.session.commit()
        
        predictions = Prediction.query.all()
        
        return render_template('index.html', predictions=predictions)

@app.route('/delete/<int:id>', methods=['POST'])
def delete_prediction(id):
    prediction_to_delete = Prediction.query.get_or_404(id)
    db.session.delete(prediction_to_delete)
    db.session.commit()
    
    predictions = Prediction.query.all()
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
