import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import requests

app = dash.Dash(__name__)

# Define some basic styles
input_style = {'margin-bottom': '10px', 'width': '100%', 'padding': '8px', 'font-size': '16px'}
label_style = {'font-weight': 'bold', 'margin-top': '10px', 'display': 'block'}
button_style = {'background-color': '#4CAF50', 'color': 'white', 'padding': '10px 24px', 'font-size': '16px', 'border': 'none', 'border-radius': '5px', 'cursor': 'pointer'}
header_style = {'text-align': 'center', 'color': '#333', 'font-family': 'Arial, sans-serif'}

app.layout = html.Div([
    html.H1("Maintenance Issue Prediction Dashboard", style=header_style),
    
    # Input fields with styling
    html.Div([
        html.Div([
            html.Label("Issue Description", style=label_style),
            dcc.Input(id='issue_desc', type='text', value='', placeholder='Enter issue description', style=input_style),
        ], style={'margin-bottom': '20px'}),
        
        html.Div([
            html.Label("Severity (1-10)", style=label_style),
            dcc.Input(id='severity', type='number', value=4, min=1, max=10, style=input_style),
        ], style={'margin-bottom': '20px'}),
        
        html.Div([
            html.Label("Total Downtime (hours)", style=label_style),
            dcc.Input(id='downtime', type='number', value=1, min=0, style=input_style),
        ], style={'margin-bottom': '20px'}),
        
        html.Div([
            html.Label("OEE (0 to 1)", style=label_style),
            dcc.Input(id='oee', type='number', value=0.75, min=0, max=1, step=0.01, style=input_style),
        ], style={'margin-bottom': '20px'}),
        
        html.Button('Submit', id='submit-val', n_clicks=0, style=button_style)
    ], style={'width': '40%', 'margin': 'auto'}),
    
    # Output for prediction and recommendation
    html.Div(id='prediction-output', style={'margin-top': '30px', 'text-align': 'center', 'font-size': '18px'}),
    
    # Charts and insights
    html.Div([
        dcc.Graph(id='issue-frequency-chart'),
    ], style={'width': '70%', 'margin': 'auto', 'margin-top': '40px'})
])

@app.callback(
    [Output('prediction-output', 'children'),
     Output('issue-frequency-chart', 'figure')],
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
            return ["Error: Issue description is required"], {}

        # Prepare data for API request
        input_data = {
            "description": issue_desc,
            "severity": severity,
            "total_downtime": downtime,
            "oee": oee,
            "issue_frequency": 5  # Example frequency
        }

        # Send request to FastAPI
        try:
            response = requests.post('http://127.0.0.1:8000/predict/', json=input_data)
            result = response.json()

            # Format prediction and recommendation output
            prediction_output = [
                html.H2(f"Issue Description: {issue_desc}"),
                html.P(f"Predicted Time to Fix: {result['predicted_time']:.2f} hours"),
                html.P(f"Weighted Time: {result['weighted_time']:.2f} hours"),
                html.P(f"Recommendation: {result['recommended_solution']}")
            ]

            # Example chart with dummy data
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

            return prediction_output, fig
        
        except requests.exceptions.RequestException as e:
            return [f"Error: {e}"], {}
    
    return "", {}

if __name__ == '__main__':
    app.run_server(debug=True)
