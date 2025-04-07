import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
from model import train_model, model_features, cat_features
from data_loader import load_filtered_sample
from utils import load_airport_coords
import numpy as np

# === Load + train ===
df = load_filtered_sample()
model, X = train_model(df)
airport_coords = load_airport_coords()
starting_airports = sorted(df['startingAirport'].unique())

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Flight Fare Estimator ✈️"

max_days = int(df['daysUntilFlight'].max())

app.layout = html.Div([
    html.H2("✈️ Flight Fare Estimator"),
    dcc.Dropdown(id='starting-airport', options=[{'label': a, 'value': a} for a in starting_airports], placeholder="Select Starting Airport"),
    dcc.Dropdown(id='destination-airport', placeholder="Select Destination"),
    dcc.Dropdown(id='airline', placeholder="Select Airline"),
    dcc.Dropdown(id='cabin', placeholder="Select Cabin Class"),
    html.Div([
        html.H4("Days Until Flight", style={"marginBottom": "10px", "textAlign": "center"})
    ], style={"margin": "40px 0", "padding": "0 40px"}),
    dcc.Slider(id='days-until-flight', min=0, max=max_days, step=1, value=30, marks={i: str(i) for i in range(0, max_days + 1, 30)}, tooltip={"placement": "bottom", "always_visible": True}),
    html.Button("Predict Fare", id='predict-btn', n_clicks=0),
    html.H3(id='fare-output'),
    dcc.Graph(id='route-map'),
    html.H3("Feature Importance"),
    dcc.Graph(id='importance-heatmap')
])

@app.callback(Output('destination-airport', 'options'), Input('starting-airport', 'value'))
def update_destinations(start):
    if not start:
        return []
    dests = df[df['startingAirport'] == start]['destinationAirport'].value_counts()
    return [{'label': d, 'value': d} for d in dests[dests >= 10].index]

@app.callback(Output('airline', 'options'), Input('starting-airport', 'value'), Input('destination-airport', 'value'))
def update_airlines(start, dest):
    if not start or not dest:
        return []
    options = df[(df['startingAirport'] == start) & (df['destinationAirport'] == dest)]['segmentsAirlineName'].value_counts()
    return [{'label': a, 'value': a} for a in options[options >= 10].index]

@app.callback(Output('cabin', 'options'), Input('starting-airport', 'value'), Input('destination-airport', 'value'), Input('airline', 'value'))
def update_cabins(start, dest, airline):
    if not all([start, dest, airline]):
        return []
    mask = (df['startingAirport'] == start) & (df['destinationAirport'] == dest) & (df['segmentsAirlineName'] == airline)
    return [{'label': c, 'value': c} for c in sorted(df[mask]['segmentsCabinCode'].unique())]

@app.callback(
    Output('fare-output', 'children'),
    Output('route-map', 'figure'),
    Input('predict-btn', 'n_clicks'),
    State('starting-airport', 'value'),
    State('destination-airport', 'value'),
    State('airline', 'value'),
    State('cabin', 'value'),
    State('days-until-flight', 'value')
)

def predict_fare(n, start, dest, airline, cabin, days):
    if None in [start, dest, airline, cabin, days]:
        return "Please complete all fields.", go.Figure()

    row = pd.DataFrame({
        'segmentsAirlineName': [airline],
        'startingAirport': [start],
        'destinationAirport': [dest],
        'segmentsCabinCode': [cabin],
        'daysUntilFlight': [days]
    })
    for col in cat_features:
        row[col] = pd.Categorical(row[col], categories=X[col].cat.categories)

    pred = model.predict(row[model_features])[0]
    pred = np.clip(pred, 50, 1000)

    start_coords = airport_coords.get(start)
    dest_coords = airport_coords.get(dest)

    if start_coords and dest_coords:
        fig = go.Figure(go.Scattergeo(
            lon=[start_coords['Longitude'], dest_coords['Longitude']],
            lat=[start_coords['Latitude'], dest_coords['Latitude']],
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2, color='blue')
        ))
        fig.update_layout(geo=dict(scope='usa'), title=f"{start} → {dest} Route")
    else:
        fig = go.Figure()

    return f"Predicted Fare: ${pred:.2f}", fig

@app.callback(Output('importance-heatmap', 'figure'), Input('predict-btn', 'n_clicks'))
def update_importance_heatmap(n):
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': model_features, 'Importance': importances}).sort_values(by='Importance')
    fig = go.Figure(go.Bar(x=df_imp['Importance'], y=df_imp['Feature'], orientation='h',
                           marker=dict(color=df_imp['Importance'], colorscale='YlOrRd')))
    fig.update_layout(title="🔍 Feature Importance", height=800, margin=dict(l=150))
    return fig

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # fallback if PORT isn't set
    app.run(host="0.0.0.0", port=port, debug=True)
