# Create API of ML model using flask
'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import MinMaxScaler
import config
import requests
from sklearn.preprocessing import minmax_scale
import pandas as pd



app = Flask(__name__)

# Load the model
model = pickle.load(open('./model/sf_reduced_crime_model_knn.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    if request.method == "POST":
        #data = request.get_json(force=True)
        print(request.form['weekday'])
        weekday = float(request.form["weekday"])
        month = float(request.form['month'])
        time = float(request.form['time'])
        police_district = float(request.form['pd'])
        address = request.form['address']

        response = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={config.key}')
        #print(response.json())
        location = response.json()

        latitude = location["results"][0]['geometry']['location']['lat']
        longitude = location["results"][0]['geometry']['location']['lng']

        df = pd.read_csv('final_encoded_sf_police_data.csv')
        # print(df)
        features_names = ['incident_day_of_week', 'incident_month',
                          'incident_time', 'police_district', 'longitude', 'latitude']


        X = df[features_names]
        live = pd.DataFrame([[weekday, month, time, police_district, longitude, latitude]],
                            columns=features_names)
        live_df = pd.concat([X, live], ignore_index=True)

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(live_df)
        data = [X_train_scaled[-1]]
        print(data)

        #data = float(request.form['exp'])
        #print("Data", model.predict([[data]]))
        # Make prediction using model loaded from disk as per the data.
        #scaler = MinMaxScaler()
        #X_train_scaled = scaler.fit_transform(data)
        #print(X_train_scaled)
        
        #print(minmax_scale(data))
        #X_train_scaled = scaler.fit_transform(X_train)
        prediction = model.predict(data)
        print(prediction)

        # Take the first value of prediction
        output = prediction[0]
        result = ""
        gif = ""
        if(output == 3):
            result = "Vehicle Related Crime"
            gif = "https://media.giphy.com/media/Ss0HSjMmirJ5hhPBDn/giphy.gif"
        elif(output == 2):
            result = "Vandalism"
            gif = "https://media.giphy.com/media/3o6Mb31TPg95otUaoU/giphy.gif"
        else: 
            result = "Assault"
            gif = "https://media.giphy.com/media/18kPwV9qSCY9O/giphy.gif"
        # vandalism(2), vehicle-related-crime(3), assault(0)
        return render_template("results.html", result=result, address=address, gif=gif)


if __name__ == '__main__':
    app.run(debug=True)
