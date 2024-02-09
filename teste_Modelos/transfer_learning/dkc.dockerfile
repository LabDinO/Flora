# Use the official Ubuntu base image
FROM ubuntu:latest

# Set the working directory
WORKDIR /app

# Copie o arquivo tar para o contêiner
COPY Python-3.7.11.tgz  /app/Python-3.7.11.tgz

# Install necessary dependencies an python
RUN apt-get update -y && \
    apt-get install -y build-essential libssl-dev zlib1g-dev && \
    tar -xzvf Python-3.7.11.tgz && \
    cd /app/Python-3.7.11 && \
    ./configure && \
    make && \
    make install

# Copy the Python script and data directory into the container
COPY transfer_learning_tutorial_Resnet_Hymenoptera.py /app/transfer_learning_tutorial_Resnet_Hymenoptera.py
COPY hymenoptera_data /app/hymenoptera_data


# Install Python dependencies
RUN pip3 install torch torchvision matplotlib pillow


# Set the default command to run your script
CMD ["python3", "transfer_learning_tutorial_Resnet_Hymenoptera.py"]
