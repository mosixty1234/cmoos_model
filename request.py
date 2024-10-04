import requests

# Define the API URL (adjust the port if needed)
url = "http://127.0.0.1:8000/predict/"

# Sample input data to be sent to the model
sample_data = {
    "description": "Pump failure due to overheating",
    "severity": 7,
    "occurrence": 4,
    "detection": 3
}

# Send a POST request to the model
response = requests.post(url, json=sample_data)

# Print the status code and the JSON response (if any)
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
