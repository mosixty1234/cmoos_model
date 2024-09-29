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

# Define styles
input_style = {'margin-bottom': '10px', 'width': '100%', 'padding': '8px', 'font-size': '16px'}
label_style = {'font-weight': 'bold', 'margin-top': '10px', 'display': 'block'}
button_style = {
    'background-color': '#007BFF', 'color': 'white', 'padding': '10px 24px', 'font-size': '16px',
    'border': 'none', 'border-radius': '5px', 'cursor': 'pointer', 'width': '100%', 'transition': 'background-color 0.3s ease'
}
icon_style = {'margin-left': '10px', 'color': 'orange'}
header_style = {'text-align': 'center', 'color': '#333', 'font-family': 'Arial, sans-serif'}
error_message_style = {'margin-top': '20px', 'text-align': 'center', 'font-size': '18px', 'color': 'red'}
success_message_style = {'margin-top': '20px', 'text-align': 'center', 'font-size': '18px', 'color': 'green'}

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("Maintenance Issue Prediction Dashboard", style=header_style),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Issue Description", style=label_style),
                        dcc.Input(id='issue_desc', type='text', value='', placeholder='Enter issue description', style=input_style),

                        html.Label("Severity (1-10)", style=label_style),
                        dcc.Input(id='severity', type='number', value=4, min=1, max=10, style=input_style),
                        html.Div(id='severity-icon', style={'display': 'inline-block'}),

                        html.Label("Total Downtime (hours)", style=label_style),
                        dcc.Input(id='downtime', type='number', value=1, min=0, style=input_style),

                        html.Label("OEE (0 to 1)", style=label_style),
                        dcc.Input(id='oee', type='number', value=0.75, min=0, max=1, step=0.01, style=input_style),

                        html.Button('Submit', id='submit-val', n_clicks=0, style=button_style, className='hover-button'),
                    ],
                    width=6,
                )
            ],
            justify='center',
            className='mt-5'
        ),

        html.Div(id='status-message', style={'margin-top': '20px', 'text-align': 'center', 'font-size': '18px'}),

        dcc.Loading(
            id="loading",
            type="circle",
            children=html.Div(id='prediction-output', style={'margin-top': '30px', 'text-align': 'center', 'font-size': '18px'})
        ),

        dbc.Row(
            dbc.Col(
                dcc.Graph(id='issue-frequency-chart'),
                width=8
            ),
            justify='center',
            className='mt-5'
        )
    ],
    fluid=True,
)

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
        # Validate input fields
        if not issue_desc:
            return ["Error: Issue description is required"], {}, html.I(className="fa fa-question-circle", style=icon_style), "Please enter an issue description."

        if not (1 <= severity <= 10):
            return ["Error: Severity must be between 1 and 10"], {}, html.I(className="fa fa-exclamation-triangle", style={'color': 'red'}), "Severity must be between 1 and 10."

        if not (0 <= oee <= 1):
            return ["Error: OEE must be between 0 and 1"], {}, html.I(className="fa fa-exclamation-triangle", style={'color': 'red'}), "OEE must be between 0 and 1."

        severity_icon = html.I(className="fa fa-exclamation-triangle", style=icon_style) if severity and 1 <= severity <= 10 else html.I(className="fa fa-question-circle", style={'color': 'gray'})

        input_data = {
            "description": issue_desc,
            "severity": severity,
            "total_downtime": downtime,
            "oee": oee,
            "issue_frequency": 5  # Example frequency; consider making this dynamic or configurable
        }

        try:
            # Send request to the API
            response = requests.post(API_URL, json=input_data, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses

            result = response.json()
            if 'predicted_time' in result and 'weighted_time' in result and 'recommended_solution' in result:
                prediction_output = [
                    html.H2(f"Issue Description: {issue_desc}"),
                    html.P(f"Predicted Time to Fix: {result['predicted_time']:.2f} hours"),
                    html.P(f"Weighted Time: {result['weighted_time']:.2f} hours"),
                    html.P(f"Recommendation: {result['recommended_solution']}")
                ]

                # Dynamic issue frequency data (for demonstration, you can replace it with actual data)
                issue_data = pd.DataFrame({
                    'Issue': ['Motor Failure', 'Pump Malfunction', 'Sensor Issue', 'Overheating'],
                    'Frequency': [10, 7, 5, 3]  # Example frequencies; replace with dynamic data if available
                })

                fig = px.bar(issue_data, x='Issue', y='Frequency', title="Issue Frequency",
                             text='Frequency', labels={'Issue': 'Type of Issue', 'Frequency': 'Occurrences'},
                             color='Frequency', color_continuous_scale=px.colors.sequential.Teal)

                fig.update_layout(
                    title_font_size=24,
                    xaxis_title="Issue Type",
                    yaxis_title="Number of Occurrences",
                    uniformtext_minsize=10,
                    uniformtext_mode='hide',
                    margin=dict(l=50, r=50, t=80, b=50),
                    template='plotly_dark'  # Update for dark theme
                )

                fig.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=0.8, textposition='outside')

                return prediction_output, fig, severity_icon, "Prediction successful!"
            else:
                error_message = result.get('detail', 'Unexpected response structure from API.')
                return [f"Error: {error_message}"], {}, severity_icon, error_message

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return ["Error: Unable to process your request. Please try again later."], {}, severity_icon, "HTTP error occurred."
        except requests.exceptions.Timeout:
            logger.error("The request timed out")
            return ["Error: Request timed out. Please try again later."], {}, severity_icon, "Request timed out."
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return ["Error: Unable to reach the server. Please try again later."], {}, severity_icon, "Server connection error."

    return [], {}, html.I(className="fa fa-question-circle", style=icon_style), ""

if __name__ == '__main__':
    app.run_server(debug=True)
