
mport numpy as np
import pickle
import pandas as pd
from flask import Flask, request

app=Flask(__name__)
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


@app.route('/')
def hello():
    return "Welcome All to Week-14"


@app.route('/predict')
def predict_class():
    sepal_length=request.args.get('sepal_length')
    sepal_width=request.args.get('sepal_width')
    petal_length=request.args.get('petal_length')
    petal_width=request.args.get('petal_width')
    prediction=classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    return " The Predicated Class is"+ str(prediction)


if __name__=='__main__':
    app.run()
                             