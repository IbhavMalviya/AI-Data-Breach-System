# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy everything to container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Streamlit uses
EXPOSE 8501

# ✅ Proper JSON-style CMD to prevent shell issues
CMD ["streamlit", "run", "App/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
