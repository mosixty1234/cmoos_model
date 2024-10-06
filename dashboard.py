import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import requests
import dash_bootstrap_components as dbc
import logging

# API URL configuration
API_URL = "http://127.0.0.1:8000/predict/"

# Initialize logging
logging.basicConfig(filename='dash_app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define styles
input_style = {
    'margin-bottom': '10px', 
    'width': '100%', 
    'padding': '10px', 
    'font-size': '16px', 
    'border-radius': '8px', 
    'background-color': '#444',  # Dark grey background for inputs
    'color': 'white', 
    'border': '1px solid #555'   # Slightly lighter grey border
}
label_style = {
    'font-weight': 'bold', 
    'margin-top': '10px', 
    'display': 'block', 
    'color': 'white'
}
button_style = {
    'background-color': '#007BFF', 
    'color': 'white', 
    'padding': '12px 28px', 
    'font-size': '16px', 
    'border': 'none',
    'border-radius': '12px', 
    'cursor': 'pointer', 
    'width': '100%', 
    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
    'transition': 'background-color 0.3s ease, transform 0.2s ease', 
    'text-transform': 'uppercase'
}
header_style = {
    'text-align': 'center', 
    'color': 'white', 
    'font-family': 'Verdana, sans-serif', 
    'margin-top': '20px'
}
container_style = {
    'background': 'linear-gradient(145deg, #1c1c1c, #444)', 
    'border-radius': '15px', 
    'padding': '20px'
}

# App layout
app.layout = dbc.Container(
    [
        html.H1("✨ CMOOS Maintenance Issue Prediction Dashboard ✨", style=header_style),

        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Label("Issue Description", style=label_style),
                    dcc.Input(id='issue_desc', type='text', placeholder='Enter issue description', style=input_style),
                    html.Label("Severity (1-10)", style=label_style),
                    dcc.Input(id='severity', type='number', min=1, max=10, value=4, style=input_style),
                    html.Label("Occurrence (1-10)", style=label_style),
                    dcc.Input(id='occurrence', type='number', min=1, max=10, value=4, style=input_style),
                    html.Label("Detection (1-10)", style=label_style),
                    dcc.Input(id='detection', type='number', min=1, max=10, value=4, style=input_style),
                    html.Label("Total Downtime (hours)", style=label_style),
                    dcc.Input(id='downtime', type='number', min=0, value=1, style=input_style),
                    html.Label("Calculated RPN (1 to 1000)", style=label_style),
                    html.Div(id='calculated_rpn', style={'font-weight': 'bold', 'margin-top': '10px', 'color': 'white'}),
                    html.Button('Submit prediction', id='submit-val', n_clicks=0, style=button_style),
                ], style=container_style),
                width=6
            ),
            justify='center',
            className='mt-5'
        ),

        html.Div(id='status-message', style={'margin-top': '20px', 'text-align': 'center', 'font-size': '18px', 'color': 'white'}),
        dcc.Loading(id="loading", type="circle", children=html.Div(id='prediction-output', style={'margin-top': '30px', 'text-align': 'center', 'font-size': '18px', 'color': 'white'})),

        dbc.Row(
            dbc.Col(dcc.Graph(id='issue-frequency-chart'), width=8),
            justify='center',
            className='mt-5'
        )
    ],
    fluid=True,
    style={'backgroundColor': 'black', 'minHeight': '100vh', 'color': 'white'}  # Set the overall background to black
)

# Helper function to convert predicted time to hours or minutes
def format_time(predicted_time: float) -> str:
    if predicted_time < 1:
        # Convert hours to minutes if less than 1 hour
        minutes = predicted_time * 60
        return f"{round(minutes, 2)} minutes"
    else:
        # Keep time in hours if it's 1 hour or more
        return f"{round(predicted_time, 2)} hours"

@app.callback(
    [Output('calculated_rpn', 'children'),
     Output('prediction-output', 'children'),
     Output('issue-frequency-chart', 'figure'),
     Output('status-message', 'children')],
    [Input('submit-val', 'n_clicks')],
    [State('severity', 'value'), State('occurrence', 'value'), State('detection', 'value'),
     State('issue_desc', 'value'), State('downtime', 'value')]
)
def update_output(n_clicks_submit, severity, occurrence, detection, issue_desc, downtime):
    # Calculate RPN
    rpn = severity * occurrence * detection
    rpn = max(1, min(rpn, 1000))  # Cap RPN between 1 and 1000

    rpn_display = f"RPN = {rpn}"  # Updated display format for RPN

    if n_clicks_submit > 0:
        logger.info(f"User submitted issue: {issue_desc}, Severity: {severity}, Occurrence: {occurrence}, Detection: {detection}, Downtime: {downtime}, RPN: {rpn}")

        # Validate inputs
        if not issue_desc:
            logger.warning("Issue description is required.")
            return rpn_display, ["Error: Issue description is required"], {}, "Please enter an issue description."
        if not (1 <= severity <= 10 and 1 <= occurrence <= 10 and 1 <= detection <= 10):
            logger.warning("Severity, Occurrence, and Detection must be between 1 and 10.")
            return rpn_display, ["Error: Severity, Occurrence, and Detection must be between 1 and 10"], {}, "Values must be between 1 and 10."

        input_data = {
            "description": issue_desc,
            "severity": severity,
            "occurrence": occurrence,
            "detection": detection,
            "total_downtime": downtime
        }
        
        try:
            response = requests.post(API_URL, json=input_data, timeout=10)
            response.raise_for_status()
            result = response.json()

            if 'predicted_time' in result and 'weighted_time' in result and 'recommended_solution' in result:
                # Extract the numeric part of the predicted time and weighted time
                predicted_time_str = result['predicted_time']
                weighted_time_str = result['weighted_time']

                # Function to extract numeric value from the string
                def extract_time(time_str):
                    return float(time_str.split()[0])  # Split the string and take the first part

                # Convert to float before formatting
                predicted_time_float = extract_time(predicted_time_str)
                weighted_time_float = extract_time(weighted_time_str)

                # Format times using the helper function
                formatted_predicted_time = format_time(predicted_time_float)
                formatted_weighted_time = format_time(weighted_time_float)

                prediction_output = [
                    html.H2(f"Issue Description: {issue_desc}", style={'color': 'white'}),
                    html.P(f"Predicted Time to Fix: {formatted_predicted_time}", style={'color': 'white'}),
                    html.P(f"Weighted Time: {formatted_weighted_time}", style={'color': 'white'}),
                    html.P(f"Recommendation: {result['recommended_solution']}", style={'color': 'white'})
                ]

                # Prepare mock data for issue frequency chart
                issue_data = pd.DataFrame({
                    'Issue': ['Motor Failure', 'Pump Malfunction', 'Sensor Issue', 'Overheating'],
                    'Frequency': [10, 7, 5, 3]
                })

                fig = px.bar(issue_data, x='Issue', y='Frequency', title="Issue Frequency",
                             text='Frequency', labels={'Issue': 'Type of Issue', 'Frequency': 'Occurrences'},
                             color='Frequency', color_continuous_scale=px.colors.sequential.Turbo)

                fig.update_layout(
                    title_font_size=24,
                    xaxis_title="Issue Type",
                    yaxis_title="Number of Occurrences",
                    plot_bgcolor="rgba(240, 240, 240, 0.8)",
                    paper_bgcolor="rgba(240, 240, 240, 0.8)",
                    font=dict(color="black")
                )

                return rpn_display, prediction_output, fig, ""
            else:
                logger.error(f"Unexpected response structure: {result}")
                return rpn_display, ["Error: Unexpected response from API"], {}, "API response format is not as expected."
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return rpn_display, ["Error: API request failed"], {}, "Please check your API server."

    return rpn_display, "", {}, ""

if __name__ == '__main__':
    app.run_server(debug=True)
