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
app.title = "Flightwise"

max_days = int(df['daysUntilFlight'].max())
app.layout = html.Div([
    html.Div([
        html.H1("Flightwise", style={
            "textAlign": "center",
            "fontSize": "48px",
            "marginBottom": "10px",
            "color": "#003366"
        }),
        html.H3("Smarter flight booking, backed by data.", style={
            "textAlign": "center",
            "color": "#666",
            "marginBottom": "30px"
        }),
        html.H5("Developed by Jared Samson", style={
            "textAlign": "center",
            "color": "#999",
            "marginBottom": "40px"
        })
    ]),

    html.Div([

        html.Div([
            html.Label("Starting Airport", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id='starting-airport',
                options=[{'label': a, 'value': a} for a in starting_airports],
                placeholder="Select Starting Airport"
            )
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Destination Airport", style={"fontWeight": "bold"}),
            dcc.Dropdown(id='destination-airport', placeholder="Select Destination")
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Airline", style={"fontWeight": "bold"}),
            dcc.Dropdown(id='airline', placeholder="Select Airline")
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Cabin Class", style={"fontWeight": "bold"}),
            dcc.Dropdown(id='cabin', placeholder="Select Cabin Class")
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Days Until Flight", style={
                "fontWeight": "bold",
                "textAlign": "center",
                "marginBottom": "10px"
            }),
            dcc.Slider(
                id='days-until-flight',
                min=0,
                max=max_days,
                step=1,
                value=30,
                marks={i: str(i) for i in range(0, max_days + 1, 30)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={"margin": "40px 0"}),

        html.Button("Predict Fare", id='predict-btn', n_clicks=0, style={
            "width": "100%",
            "padding": "12px",
            "fontSize": "18px",
            "backgroundColor": "#0066cc",
            "color": "white",
            "border": "none",
            "borderRadius": "6px",
            "cursor": "pointer"
        })

    ], style={
        "maxWidth": "600px",
        "margin": "0 auto",
        "padding": "30px",
        "backgroundColor": "#fefefe",
        "borderRadius": "12px",
        "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.1)"
    }),

    html.Div([
        html.H3(id='fare-output', style={"textAlign": "center", "marginTop": "40px"}),
        dcc.Graph(id='route-map', style={"marginTop": "20px"}),
        html.H3("Feature Importance", style={"textAlign": "center", "marginTop": "40px"}),
        dcc.Graph(id='importance-heatmap')
    ], style={
        "maxWidth": "900px",
        "margin": "0 auto",
        "padding": "40px 20px"
    })
], style={"backgroundColor": "#f0f4f8", "fontFamily": "Segoe UI, sans-serif", "paddingBottom": "60px"})


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
    
    from datetime import datetime, timedelta

    flight_date = datetime.today() + timedelta(days=days)
    day_of_week = flight_date.weekday()  # Monday=0, Sunday=6

    row = pd.DataFrame({
        'segmentsAirlineName': [airline],
        'startingAirport': [start],
        'destinationAirport': [dest],
        'segmentsCabinCode': [cabin],
        'daysUntilFlight': [days],
        'dayOfWeek': [day_of_week]
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
