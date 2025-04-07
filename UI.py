import dash
from dash import dcc, html

app = dash.Dash(__name__)
app.title = "Flight Fare Estimator ✈️"

app.layout = html.Div([
    html.H3("Flight Fare Estimator"),
    
    dcc.Dropdown(id='starting-airport', options=[...], placeholder="Select Starting Airport"),
    dcc.Dropdown(id='destination-airport', placeholder="Select Destination"),
    dcc.Dropdown(id='airline', placeholder="Select Airline"),
    dcc.Dropdown(id='cabin', placeholder="Select Cabin Class"),
    
    dcc.Slider(id='days-until-flight', min=0, max=180, value=30),
    dcc.Slider(id='seats-remaining', min=1, max=30, value=5),
    
    html.Button("Predict Fare", id='predict-btn', n_clicks=0),
    
    html.Div(id='fare-output'),
    
    dcc.Graph(id='route-map')
])

if __name__ == '__main__':
    app.run_server(debug=True)
