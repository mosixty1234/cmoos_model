import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import dash_daq as daq

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Maintenance Issue Prediction"),
    
    dcc.Input(id='issue-description', type='text', placeholder='Enter issue description'),
    html.Div(id='description-status', style={'color': 'orange'}),
    
    dcc.Input(id='severity', type='number', placeholder='Enter severity (1-10)'),
    html.Div(id='severity-icon', style={'display': 'inline-block'}),
    
    dcc.Input(id='downtime', type='number', placeholder='Enter downtime (in minutes)'),
    
    dcc.Input(id='oee', type='number', placeholder='Enter OEE (0.0 - 1.0)'),
    
    html.Button('Predict', id='predict-button'),
    
    dcc.Loading(
        id="loading",
        type="circle",
        children=[html.Div(id='prediction-output')]
    ),
    
    html.Div(id='recommendation-output', style={'margin-top': '20px'})
])

@app.callback(
    [Output('prediction-output', 'children'),
     Output('recommendation-output', 'children'),
     Output('severity-icon', 'children'),
     Output('description-status', 'children')],
    [Input('predict-button', 'n_clicks')],
    [Input('issue-description', 'value'),
     Input('severity', 'value'),
     Input('downtime', 'value'),
     Input('oee', 'value')]
)
def update_prediction(n_clicks, issue_description, severity, downtime, oee):
    if n_clicks is None:
        return '', '', '', ''
    
    description_status = "Fetching results..."
    
    if severity is not None and 1 <= severity <= 10:
        severity_icon = html.I(className="fa fa-exclamation-triangle", style={'color': 'orange'})
    else:
        severity_icon = html.I(className="fa fa-question-circle", style={'color': 'gray'})
    
    # Simulate a request to the API
    try:
        response = requests.post('http://127.0.0.1:8000/predict/', json={
            'description': issue_description,
            'severity': severity,
            'downtime': downtime,
            'oee': oee
        })
        response.raise_for_status()  # Raise an error for bad responses
        prediction = response.json()['prediction']
        recommendation = response.json()['recommendation']
        
        description_status = "Prediction successful!"
        
        return f"Predicted Time to Fix: {prediction:.2f} hours", recommendation, severity_icon, description_status
    
    except Exception as e:
        description_status = "Error fetching results"
        return '', '', '', description_status

if __name__ == '__main__':
    app.run_server(debug=True)
