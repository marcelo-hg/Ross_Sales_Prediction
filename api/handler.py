import pandas as pd
from flask import Flask, request, Response
from rossman.Rossman import Rossman
import pickle

# Loading model
with open('D:/Marcelo/Python Scripts/Ross_Sales/model/model_rossman_tuned.pkl', 'rb') as f:
   model = pickle.load(f)   

app = Flask( __name__ )

@app.route( '/Ross_Sales/predict', methods=['POST'] )
def rossman_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        
        if isinstance( test_json, dict ): # unique example 
            test_raw = pd.DataFrame( test_json, index=[0] )
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossman class
        pipeline = Rossman()
        
        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        
        # prediction
        pipeline.get_prediction( model, test_raw, df3 )
        
        
    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    app.run( '0.0.0.0', port=5000 )