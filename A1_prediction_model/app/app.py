import dash
from dash import dcc, html
import pandas as pd
import pickle
import numpy as np
from dash.dependencies import Input, Output

# Load the saved model and scaler
with open("model/random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("model/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Car Price Prediction App"),
    html.Label("Year:"),
    dcc.Input(id='year', type='number', value=2018, step=1),

    html.Label("Km Driven:"),
    dcc.Input(id='km_driven', type='number', value=50000, step=1000),

    html.Label("Max Power:"),
    dcc.Input(id='max_power', type='number', value=110, step=1),

    html.Button('Predict', id='predict-btn', n_clicks=0),
    html.H3("Predicted Selling Price:"),
    html.Div(id='output-prediction')
])

@app.callback(
    Output('output-prediction', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('year', 'value'),
    Input('km_driven', 'value'),
    Input('max_power', 'value')
)
def predict(n_clicks, year, km_driven, max_power):
    if n_clicks > 0:
        # Create DataFrame
        new_data = pd.DataFrame([[year, km_driven, max_power]], columns=['year', 'km_driven', 'max_power'])

        # Scale input
        new_data_scaled = scaler.transform(new_data)

        # Predict
        predicted_price_log = model.predict(new_data_scaled)
        predicted_price = np.expm1(predicted_price_log)

        return f"Predicted Selling Price: {predicted_price[0]:,.2f} lakhs"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")