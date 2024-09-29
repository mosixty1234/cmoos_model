import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import requests
import os
import logging

# API URL configuration (use environment variables for production)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict/")

# Define styles
input_style = {'margin-bottom': '10px', 'width': '100%', 'padding': '8px', 'font-size': '16px'}
label_style = {'font-weight': 'bold', 'margin-top': '10px', 'display': 'block'}
button_style = {'background-color': '#4CAF50', 'color': 'white', 'padding': '10px 24px', 'font-size': '16px', 'border': 'none', 'border-radius': '5px', 'cursor': 'pointer'}
header_style = {'text-align': 'center', 'color': '#333', 'font-family': 'Arial, sans-serif'}
icon_style = {'margin-left': '10px', 'color': 'orange'}

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Maintenance Issue Prediction Dashboard", style=header_style),
    
    # Input fields with styling
    html.Div([
        html.Label("Issue Description", style=label_style),
        dcc.Input(id='issue_desc', type='text', value='', placeholder='Enter issue description', style=input_style),
        
        html.Label("Severity (1-10)", style=label_style),
        dcc.Input(id='severity', type='number', value=4, min=1, max=10, style=input_style),
        html.Div(id='severity-icon', style={'display': 'inline-block'}),
        
        html.Label("Total Downtime (hours)", style=label_style),
        dcc.Input(id='downtime', type='number', value=1, min=0, style=input_style),
        
        html.Label("OEE (0 to 1)", style=label_style),
        dcc.Input(id='oee', type='number', value=0.75, min=0, max=1, step=0.01, style=input_style),
        
        html.Button('Submit', id='submit-val', n_clicks=0, style=button_style)
    ], style={'width': '40%', 'margin': 'auto'}),
    
    # Status message area
    html.Div(id='status-message', style={'margin-top': '20px', 'text-align': 'center', 'font-size': '18px', 'color': 'red'}),
    
    # Output for prediction and recommendation
    dcc.Loading(id="loading", type="circle", children=html.Div(id='prediction-output', style={'margin-top': '30px', 'text-align': 'center', 'font-size': '18px'})),
    
    # Charts and insights
    html.Div([
        dcc.Graph(id='issue-frequency-chart'),
    ], style={'width': '70%', 'margin': 'auto', 'margin-top': '40px'})
])

@app.callback(
    [Output('prediction-output', 'children'),
     Output('issue-frequency-chart', 'figure'),
     Output('severity-icon', 'children'),
     Output('status-message', 'children')],
    [Input('submit-val', 'n_clicks')],
    [State('issue_desc', 'value'),
     State('severity', 'value'),
     State('downtime', 'value'),
     State('oee', 'value')]
)
def update_output(n_clicks, issue_desc, severity, downtime, oee):
    if n_clicks > 0:
        # Check if issue description is provided
        if not issue_desc:
            return ["Error: Issue description is required"], {}, html.I(className="fa fa-question-circle", style=icon_style), "Please enter an issue description."
        
        # Validate severity
        if not (1 <= severity <= 10):
            return ["Error: Severity must be between 1 and 10"], {}, html.I(className="fa fa-exclamation-triangle", style={'color': 'red'}), "Severity must be between 1 and 10."
        
        # Validate OEE
        if not (0 <= oee <= 1):
            return ["Error: OEE must be between 0 and 1"], {}, html.I(className="fa fa-exclamation-triangle", style={'color': 'red'}), "OEE must be between 0 and 1."
        
        severity_icon = html.I(className="fa fa-exclamation-triangle", style=icon_style) if severity and 1 <= severity <= 10 else html.I(className="fa fa-question-circle", style={'color': 'gray'})

        # Prepare data for API request
        input_data = {
            "description": issue_desc,
            "severity": severity,
            "total_downtime": downtime,
            "oee": oee,
            "issue_frequency": 5  # Example frequency; consider making this dynamic or configurable
        }

        # Send request to FastAPI with a timeout for safety
        try:
            response = requests.post(API_URL, json=input_data, timeout=10)
            
            # Check if the response status is OK
            if response.status_code == 200:
                result = response.json()
                
                # Debugging: Log the entire response
                logger.debug(f"API Response Status Code: {response.status_code}")
                logger.debug(f"API Response Body: {response.text}")
                
                # Check if 'predicted_time' exists in the response
                if 'predicted_time' in result and 'weighted_time' in result and 'recommended_solution' in result:
                    # Format prediction and recommendation output
                    prediction_output = [
                        html.H2(f"Issue Description: {issue_desc}"),
                        html.P(f"Predicted Time to Fix: {result['predicted_time']:.2f} hours"),
                        html.P(f"Weighted Time: {result['weighted_time']:.2f} hours"),
                        html.P(f"Recommendation: {result['recommended_solution']}")
                    ]
                    
                    # Example chart with dynamic data (you can fetch real data from the API if available)
                    issue_data = pd.DataFrame({
                        'Issue': ['Motor Failure', 'Pump Malfunction', 'Sensor Issue', 'Overheating'],
                        'Frequency': [10, 7, 5, 3]
                    })
                    
                    fig = px.bar(issue_data, 
                                 x='Issue', 
                                 y='Frequency', 
                                 title="Issue Frequency",
                                 text='Frequency',  # Add labels on bars
                                 labels={'Issue': 'Type of Issue', 'Frequency': 'Occurrences'},
                                 color='Frequency',
                                 color_continuous_scale=px.colors.sequential.Teal)
                    
                    # Customize chart appearance
                    fig.update_layout(
                        title_font_size=24,
                        xaxis_title="Issue Type",
                        yaxis_title="Number of Occurrences",
                        uniformtext_minsize=10, 
                        uniformtext_mode='hide',
                        margin=dict(l=50, r=50, t=80, b=50),
                        template='simple_white'
                    )
                    
                    fig.update_traces(marker_line_color='black',  # Outline bars
                                      marker_line_width=1.5,      # Width of bar outlines
                                      opacity=0.8,                # Slight transparency
                                      textposition='outside')      # Position of text labels on bars
                    
                    return prediction_output, fig, severity_icon, "Prediction successful!"
                else:
                    # If expected keys are missing
                    error_message = result.get('detail', 'Unexpected response structure from API.')
                    return [f"Error: {error_message}"], {}, severity_icon, error_message
            else:
                try:
                    error_info = response.json()
                    error_message = error_info.get('detail', 'An error occurred while processing the request.')
                except ValueError:
                    error_message = 'An unknown error occurred.'
                return [f"Error: {error_message}"], {}, severity_icon, error_message

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return ["Error: Unable to reach the server. Please try again later."], {}, severity_icon, "Server connection error."
        
    return [], {}, html.I(className="fa fa-question-circle", style=icon_style), ""

if __name__ == '__main__':
    app.run_server(debug=True)
