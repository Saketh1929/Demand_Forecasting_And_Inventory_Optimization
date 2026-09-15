FROM python:3.10-slim

WORKDIR /app

# Install dependencies
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
