import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Initialize Dash app
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Car Price Prediction App"),
    
    html.Label("Year:"),
    dcc.Input(id='year', type='number', value=2018),
    
    html.Label("Kilometers Driven:"),
    dcc.Input(id='km_driven', type='number', value=50000),
    
    html.Label("Max Power:"),
    dcc.Input(id='max_power', type='number', value=110),
    
    html.Button('Predict Price', id='predict-btn', n_clicks=0),
    
    html.H3("Predicted Selling Price:"),
    html.Div(id='prediction-output')
])

# Callback to update prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('year', 'value'),
    State('km_driven', 'value'),
    State('max_power', 'value')
)
def predict_price(n_clicks, year, km_driven, max_power):
    if n_clicks > 0:
        # Create DataFrame for input
        new_data = pd.DataFrame({'year': [year], 'km_driven': [km_driven], 'max_power': [max_power]})
        
        # Scale features
        new_data_scaled = scaler.transform(new_data)
        
        # Predict
        predicted_price_log = model.predict(new_data_scaled)
        predicted_price = np.expm1(predicted_price_log)

        return f"Predicted Selling Price: {predicted_price[0]:,.2f} lakhs"
    
    return ""

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)