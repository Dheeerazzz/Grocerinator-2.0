from flask import Flask, render_template, jsonify, request
import joblib
import os
import io
import cv2
import json
import requests
import base64
import numpy as np
import csv

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

    try:
        # Get form data
        height = request.form['height']
        weight = request.form['weight']
        diseases = request.form.getlist('diseases')
        
        # Convert form values to floats
        fat = float(request.form['fat'])
        fiber = float(request.form['fiber'])
        carbs = float(request.form['carbs'])
        energy = float(request.form['energy'])
        proteins = float(request.form['proteins'])
        salts = float(request.form['salts'])
        sugars = float(request.form['sugars'])
        import csv

        # Define the data to be inserted as a list
        row_data = [fat,fiber,carbs,energy,proteins,salts,sugars]
        csv_file = 'userdata.csv'
        with open(csv_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(row_data)
        print("Row data inserted successfully!")

        body_type = request.form['body_type']
        new = []

        nutritional_values = [fat, fiber, carbs, energy, proteins, salts, sugars]
        print(nutritional_values)
        safety_score = safety_model.predict([nutritional_values])

        yes_or_no = yesorno_model.predict([nutritional_values])

        if body_type == 'skinny':
            if (round(yes_or_no[0][3])) ==0:
                new.append('This product is not suitable for your bodytype (Skinny)')
            
        elif body_type == 'fit':
            if (round(yes_or_no[0][4])) ==0:
                new.append('This product is not recommended for your bodytype.')
            
        elif body_type == 'obese':
            if (round(yes_or_no[0][5])) ==0:
                new.append('This product is not suitable for your bodytype (Obese)')
            

        if 'Diabetes' in diseases:
            if (round(yes_or_no[0][0])) ==0:
                new.append('ALERT : This product is not recommended because of high added sugar amount')
            
        if 'Blood Pressure' in diseases:
            if (round(yes_or_no[0][1])) ==0:
                new.append('ALERT : This product is not recommended because of high salt amount')
        if 'Cholesterol' in diseases:
            if (round(yes_or_no[0][0])) ==0:
                new.append('ALERT : This product is not recommended because of high amounts of fat')

        return render_template('index.html', safety_score=np.round(safety_score, decimals=1), new=new)

    except ValueError:
        # Handle error if form fields are empty or contain invalid data
        return render_template('index.html', error_message="Error: Invalid input for numeric fields")

    except Exception as e:
        # Handle other unexpected errors
        return render_template('index.html', error_message=str(e))

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

def perform_ocr(image):
    height, width, _ = image.shape
    roi = image[0: height, 100: width]

    url_api = "https://api.ocr.space/parse/image"
    _, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
    file_bytes = io.BytesIO(compressedimage)

    result = requests.post(url_api,
                           files={"file": file_bytes},
                           data={"apikey": "K83764290988957",
                                 "language": "eng"})
    result = result.content.decode()
    result = json.loads(result)

    parsed_results = result.get("ParsedResults")
    if parsed_results is not None:
        parsed_result = parsed_results[0]
        text_detected = parsed_result.get("ParsedText")
        return text_detected
    else:
        return "ParsedResults is None"


@app.route('/process_ocr', methods=['POST'])
def process_ocr():
    try:
        # Get the base64-encoded image from the request
        image_data = request.json.get('image')
        image_bytes = io.BytesIO(base64.b64decode(image_data))

        # Perform OCR on the image
        img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        ocr_result = perform_ocr(img)

        # Render the template with the OCR result
        return render_template('index.html', ocr_result=ocr_result)

    except Exception as e:
        return render_template('index.html', error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True)
