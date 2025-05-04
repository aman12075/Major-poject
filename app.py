from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load pre-trained models and preprocessors
crop_model = pickle.load(open('model.pkl', 'rb'))  # Crop recommendation model
fertilizer_model = pickle.load(open('classifier1.pkl', 'rb'))  # Fertilizer recommendation model
scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))  # Scaler for crop model
dtr_model = pickle.load(open('dtr.pkl', 'rb'))  # Crop yield prediction model
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))  # Preprocessor for crop yield model

# Home page f
@app.route('/')
def home():
    return render_template("index.html")

# Crop recommendation page
@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)
        prediction = crop_model.predict(features_scaled)
        
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 
            22: "Coffee"
        }
        
        recommended_crop = crop_dict.get(prediction[0], "No recommendation")
        return render_template('crop_recommendation.html', recommended_crop=recommended_crop)
    
    return render_template('crop_recommendation.html')

# Crop yield prediction page
@app.route('/crop-yield-prediction', methods=['GET', 'POST'])
def crop_yield_prediction():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            rainfall = float(request.form['rainfall'])
            pesticides = float(request.form['pesticides'])
            temp = float(request.form['temp'])
            area = request.form['area']
            item = request.form['item']

            features = np.array([[year, rainfall, pesticides, temp, area, item]], dtype=object)
            transformed = preprocessor.transform(features)
            result = dtr_model.predict(transformed).reshape(1, -1)

            return render_template('crop_yield_prediction.html', result=result[0])
        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('crop_yield_prediction.html')

# Fertilizer recommendation page
@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        
        prediction = fertilizer_model.predict([[N, P, K]])
        
        fertilizer_dict = {
            0: "TEN-TWENTY SIX-TWENTY SIX",
            1: "Fourteen-Thirty Five-Fourteen",
            2: "Seventeen-Seventeen-Seventeen",
            3: "TWENTY-TWENTY",
            4: "TWENTY EIGHT-TWENTY EIGHT",
            5: "DAP",
            6: "UREA"
        }
        
        recommended_fertilizer = fertilizer_dict.get(prediction[0], "No recommendation")
        return render_template('fertilizer_recommendation.html', recommended_fertilizer=recommended_fertilizer)
    
    return render_template('fertilizer_recommendation.html')

if __name__ == '__main__':
    app.run(debug=True)
