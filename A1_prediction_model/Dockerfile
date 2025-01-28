# Use official Python image
FROM python:3.13.1

# Set working directory
WORKDIR /app

# Copy files into container
COPY app/ app/
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run the application
CMD ["python", "app/app.py"]