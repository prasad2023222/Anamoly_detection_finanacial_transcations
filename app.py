#from django.shortcuts import render
from flask import Flask,request,app,jsonify,url_for,render_template
import requests
import numpy as np
import pandas as pd
import joblib

app=Flask(__name__)

loaded_obj=joblib.load("isolation_model_predict.pkl")

model=loaded_obj["model"]
processor=loaded_obj["preprocessor"]

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["Post"])
def predict_api():
    data=requests.json['data']
    input=np.array(list(data.values())).reshape(1,-1)
    new_data=processor.transform(input)
    prediction=model.predict(new_data)
    print(prediction[0])
    return jsonify(prediction[0])

@app.route("/predict",methods=["Post"])
def predict():
    data=[float(x) for x in requests.form.values()]
    final_input=processor.transform(np.array(data).reshape(1,-1))
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_txt="Anamoly datection is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)