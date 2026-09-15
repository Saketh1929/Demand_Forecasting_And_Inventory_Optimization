FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for XGBoost
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make the startup script executable
RUN chmod +x start.sh

# Railway injects the $PORT variable automatically.
EXPOSE $PORT

# Start both applications
CMD ["./start.sh"]
