# Use Python as base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install dependencies
RUN pip install dash plotly pandas numpy scikit-learn gunicorn

# Expose port 8080
EXPOSE 8080

# Command to run the app
CMD ["python", "app.py"]