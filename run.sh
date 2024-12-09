#!/bin/bash

# Navigate to the src folder relative to the base folder
cd "$(dirname "$0")/src"

echo "Running pipeline..."

# Step 1: Data Ingestion
python data_ingestion.py

# Step 2: Data Preprocessing
# Run the script to preprocess the data
python preprocessing.py 

# Step 3: Model Training and Prediction
python model_training.py
