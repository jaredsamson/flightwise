import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
from model import train_model, model_features, cat_features
from data_loader import load_filtered_sample
from utils import load_airport_coords
import numpy as np
from datetime import datetime, timedelta
import os

# === Load + train ===
df = load_filtered_sample()
model, X = train_model(df)
airport_coords = load_airport_coords()
starting_airports = sorted(df['startingAirport'].unique())

# === Dash App ===
app = dash.Dash(
    __name__,
    title="Flightwise ✈️",
    update_title="Loading flight fare model...",
    suppress_callback_exceptions=True
)

max_days = int(df['daysUntilFlight'].max())
app.layout = html.Div([
    html.Div([
        html.H1("Flightwise", style={
            "fontSize": "36px", "marginBottom": "5px", "color": "#003366",
            "fontWeight": "bold", "letterSpacing": "0.5px"
        }),
        html.H4("Smarter flight booking, backed by data.", style={
            "color": "#555", "marginBottom": "25px", "fontWeight": "400"
        }),
        html.Div("by Jared Samson", style={
            "color": "#999", "fontSize": "13px", "marginBottom": "30px"
        }),

        html.Label("Starting Airport"),
        dcc.Dropdown(
            id='starting-airport',
            options=[{'label': a, 'value': a} for a in starting_airports],
            placeholder="Select Starting Airport"
        ),

        html.Label("Destination Airport"),
        dcc.Dropdown(
            id='destination-airport',
            placeholder="Select Destination Airport"
        ),

        html.Label("Airline"),
        dcc.Dropdown(
            id='airline',
            placeholder="Select Airline"
        ),

        html.Label("Cabin Class"),
        dcc.Dropdown(
            id='cabin',
            placeholder="Select Cabin Class"
        ),

        html.Label("Days Until Flight"),
        dcc.Slider(
            id='days-until-flight',
            min=0,
            max=max_days,
            step=1,
            value=30,
            marks={i: str(i) for i in range(0, max_days + 1, 30)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Div(id='today-date-display', style={"marginTop": "10px", "color": "#555"}),
        html.Div(id='flight-date-display', style={"marginBottom": "20px", "fontWeight": "bold"}),

        html.Button("Predict Fare", id='predict-btn', n_clicks=0, style={
            "marginTop": "20px", "width": "100%", "padding": "12px", "fontSize": "16px",
            "backgroundColor": "#0066cc", "color": "white", "border": "none",
            "borderRadius": "6px", "cursor": "pointer", "transition": "0.2s"
        })
    ], style={
        "width": "340px", "padding": "30px",
        "backgroundColor": "#ffffff", "borderRight": "1px solid #ddd",
        "boxShadow": "2px 0 6px rgba(0,0,0,0.04)", "minHeight": "100vh"
    }),
        html.Div([
            html.Div(id='fare-output', style={
                "fontSize": "32px",
                "fontWeight": "700",
                "textAlign": "center",
                "color": "#003366",
                "marginTop": "30px",
                "letterSpacing": "0.5px"
            }),

            html.Div(id='route-label', style={
                "textAlign": "center",
                "fontSize": "20px",
                "fontWeight": "600",
                "color": "#444",
                "marginBottom": "10px"
            }),

            dcc.Graph(id='route-map', style={
                "height": "500px",
                "margin": "0 auto",
                "borderRadius": "8px",
                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)",
                "backgroundColor": "#fff"
            }),

            html.H4("Feature Importance", style={
                "textAlign": "center",
                "marginTop": "40px",
                "color": "#333"
            }),

            dcc.Graph(id='importance-heatmap', style={
                "height": "400px",
                "paddingBottom": "20px"
            })
        ], style={
            "flex": "1",
            "padding": "40px",
            "overflowX": "auto"
        })

    ], style={
        "display": "flex", "fontFamily": "Inter, sans-serif",
        "backgroundColor": "#f7f9fc"
    })

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
    Output('today-date-display', 'children'),
    Output('flight-date-display', 'children'),
    Input('days-until-flight', 'value')
)
def update_date_labels(days):
    today = datetime.today().date()
    flight_date = today + timedelta(days=days)
    return (
        f"Today's Date: {today.strftime('%A, %B %d, %Y')}",
        f"Flight Date: {flight_date.strftime('%A, %B %d, %Y')}"
    )

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
@app.callback(
    Output('fare-output', 'children'),
    Output('route-map', 'figure'),
    Output('route-label', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('starting-airport', 'value'),
    State('destination-airport', 'value'),
    State('airline', 'value'),
    State('cabin', 'value'),
    State('days-until-flight', 'value')
)
def predict_fare(n, start, dest, airline, cabin, days):
    if None in [start, dest, airline, cabin, days]:
        blank_fig = go.Figure()
        blank_fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': "Please complete all fields to see prediction.",
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 18}
            }]
        )
        return "Please complete all fields.", blank_fig, ""

    flight_date = datetime.today() + timedelta(days=days)
    day_of_week = flight_date.weekday()

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
            marker=dict(size=10, color='red'),
            line=dict(width=4, color='royalblue')
        ))

        fig.update_layout(
            geo=dict(
                scope='usa',
                projection_type='albers usa',
                showland=True,
                landcolor="lightgray",
                showlakes=True,
                lakecolor="lightblue"
            ),
            margin=dict(l=40, r=40, t=40, b=30),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=False
        )
    else:
        fig = go.Figure()

    route_str = f"{start} → {dest} Route"
    return f"Predicted Fare: ${pred:.2f}", fig, route_str


@app.callback(Output('importance-heatmap', 'figure'), Input('predict-btn', 'n_clicks'))
def update_importance_heatmap(n):
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': model_features, 'Importance': importances}).sort_values(by='Importance')
    fig = go.Figure(go.Bar(
        x=df_imp['Importance'],
        y=df_imp['Feature'],
        orientation='h',
        marker=dict(color=df_imp['Importance'], colorscale='YlOrRd')
    ))
    fig.update_layout(
        title="🔍 Top Influences on Predicted Fare",
        height=600,
        margin=dict(l=120, r=30, t=50, b=40),
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    return fig

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
