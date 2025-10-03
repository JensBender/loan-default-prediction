# Use an official Python runtime as a parent image
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY ./src ./src
COPY ./backend ./backend
COPY ./frontend ./frontend
COPY ./models ./models
COPY ./start.sh .

# Make the start script executable
RUN chmod +x ./start.sh

# Expose Gradio and FastAPI ports
EXPOSE 7860  
EXPOSE 8000

# Command to run the application
CMD ["./start.sh"]
