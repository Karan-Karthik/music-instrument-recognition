# Use the official Python image
FROM python:3.10.13

# Set the working directory inside the container
WORKDIR /run

# Copy all files from the current directory to the container
COPY . /run

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt update -y && apt install awscli -y

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the desired port
EXPOSE 500

# Run the run.py script
CMD ["python3", "run.py"]
