from flask import Flask, render_template, jsonify, request
import joblib
import os

app = Flask(__name__)

# Load the machine learning models
safety_model_path = os.path.join("models", "safetyscore.pkl")
yesorno_model_path = os.path.join("models", "yesorno.pkl")
safety_model = joblib.load(safety_model_path)
yesorno_model = joblib.load(yesorno_model_path)

# Define global variables for nutrients
carbs = None
fat = None
fiber = None
proteins = None
salts = None
sugars = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    global carbs, fat, fiber, proteins, salts, sugars  # Declare variables as global

    height = request.form['height']
    weight = request.form['weight']
    diseases = request.form.getlist('diseases')

    fat = float(request.form['fat'])
    fiber = float(request.form['fiber'])
    carbs = float(request.form['carbs'])  # Assign value to global variable
    energy = float(request.form['energy'])
    proteins = float(request.form['proteins'])
    salts = float(request.form['salts'])
    sugars = float(request.form['sugars'])

    body_type = request.form['body_type']
    new = []

    nutritional_values = [fat, fiber, carbs, energy, proteins, salts, sugars]
    print(nutritional_values)
    safety_score = safety_model.predict([nutritional_values])

    yes_or_no = yesorno_model.predict([nutritional_values])

    if body_type == 'skinny':
        new.append(round(yes_or_no[0][3]))
    elif body_type == 'fit':
        new.append(round(yes_or_no[0][4]))
    elif body_type == 'obese':
        new.append(round(yes_or_no[0][5]))

    if 'Diabetes' in diseases:
        new.append(round(yes_or_no[0][0]))
    if 'Blood Pressure' in diseases:
        new.append(round(yes_or_no[0][1]))
    if 'Cholesterol' in diseases:
        new.append(round(yes_or_no[0][2]))

    return render_template('index.html', safety_score=safety_score, new=new)

@app.route('/data')
def get_data():
    global carbs, fat, fiber, proteins, salts, sugars  # Access global variables

    if None in (carbs, fat, fiber, proteins, salts, sugars):
        # Return empty data if any nutrient is missing
        return jsonify({"labels": [], "values": []})

    data = {
        "labels": ['Fat', 'Fiber', 'Carbs', 'Proteins', 'Salts', 'Sugars'],
        "values": [fat, fiber, carbs, proteins, salts, sugars]
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
