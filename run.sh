#!/bin/bash

echo "Running pipeline..."

# Add commands to run your Python scripts, training models, etc.

# Step 1: Data Ingestion
python src/data_ingestion.py

# Step 2: Data Preprocessing
# Run both predictions (temperature and plant_stage) in one go
python preprocess.py temperature
python preprocess.py plant_stage


# Step 3: Model Training
python src/model_training.py
