from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load the machine learning models
safety_model_path = os.path.join("models", "safetyscore.pkl")
yesorno_model_path = os.path.join("models", "yesorno.pkl")
safety_model = joblib.load(safety_model_path)
yesorno_model = joblib.load(yesorno_model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    height = request.form['height']
    weight = request.form['weight']
    diseases = request.form.getlist('diseases')

    fat = float(request.form['fat'])
    fiber = float(request.form['fiber'])
    carbs = float(request.form['carbs'])
    energy = float(request.form['energy'])
    proteins = float(request.form['proteins'])
    salts = float(request.form['salts'])
    sugars = float(request.form['sugars'])

    body_type = request.form['body_type']
    new = []
    print("Height:", height)
    print("Weight:", weight)
    print("Selected Diseases:", diseases)

    nutritional_values = [fat, fiber, carbs, energy, proteins, salts, sugars]
    print("Nutritional Values:", nutritional_values)

    print("Body Type:", body_type)

    safety_score = safety_model.predict([nutritional_values])

    yes_or_no = yesorno_model.predict([nutritional_values])
    print(yes_or_no)
    print(type(yes_or_no))

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

if __name__ == '__main__':
    app.run(debug=True)
