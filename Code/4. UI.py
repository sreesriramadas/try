

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


@app.route('/')
def hello():
    return "Welcome All to Week-14"

@app.route('/predict', methods=["GET"])
def predict_class():
    
    """Let's predict the class for iris
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    sepal_length=request.args.get('sepal_length')
    sepal_width=request.args.get('sepal_width')
    petal_length=request.args.get('petal_length')
    petal_width=request.args.get('petal_width')
    prediction=classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    return " The Predicated Class is"+ str(prediction)

@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    
    """Let's predict the class for iris
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return " The Predicated Class for the TestFile is"+ str(list(prediction))


if __name__=='__main__':
    app.run()