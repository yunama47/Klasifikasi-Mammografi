# Base image with Python 3.10 (or any specific version you need)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install the required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port that the application will run on
# (Adjust the port as necessary; Hugging Face Spaces usually use port 7860 for Gradio apps)
EXPOSE 7860

# Command to run your application
CMD ["python", "main.py"]
