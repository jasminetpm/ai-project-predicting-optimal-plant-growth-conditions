#!/bin/bash

# Navigate to the src folder relative to the base folder
cd "$(dirname "$0")/src"

echo "Running pipeline..."

# Add commands to run your Python scripts, training models, etc.

# Step 1: Data Ingestion
python data_ingestion.py

# Step 2: Data Preprocessing
# Run both predictions (temperature and plant_stage) in one go
python preprocessing.py


# Step 3: Model Training
python model_training.py