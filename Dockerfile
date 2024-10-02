# Use the official Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /cmoos_model

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Expose the ports for FastAPI (8000) and Dash (8050)
EXPOSE 8000
EXPOSE 8050

# Start both FastAPI and Dash apps
CMD ["sh", "-c", "uvicorn app:app --host 127.0.0.1 --port 8000 & python3 dashboard.py --host 127.0.0.1 --port 8050"]