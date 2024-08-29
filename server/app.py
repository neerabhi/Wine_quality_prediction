from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load your model
model_path = os.path.join(os.path.dirname(__file__), '../model/wine_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    features = [
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
        float(request.form['density']),
        float(request.form['pH']),
        float(request.form['sulphates']),
        float(request.form['alcohol'])
    ]

    # Make a prediction
    prediction = model.predict([features])[0]
    
    # Determine the quality message
    if prediction == 0:
        quality_message = "The wine quality is not good."
    else:
        quality_message = "The wine quality is good."

    return render_template('index.html', prediction_message=quality_message)

if __name__ == '__main__':
    app.run(debug=True)
