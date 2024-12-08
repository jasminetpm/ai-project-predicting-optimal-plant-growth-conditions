import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import sys
import json

# Function to standardize the format of categories to Title Case
def standardize_format(data):
    """
    Standardize data formats by stripping ' ppm' from nutrient columns and converting them to numeric, 
    standardizing 'Plant Type' and 'Plant Stage' to Title Case, 
    and creating a new column 'Plant Type-Stage' by concatenating 'Plant Type' and 'Plant Stage'.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame after formatting standardization.
    """
    # 1. Strip " ppm" from Nutrient Sensor columns and convert to numeric
    nutrient_columns = ['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)']
    for col in nutrient_columns:
        data[col] = data[col].str.rstrip(" ppm").astype(float)
        print(f"Stripped ' ppm' and converted {col} to numeric.")
    
    # 2. Standardize 'Plant Type' and 'Plant Stage' to Title Case
    data["Plant Type"] = data["Plant Type"].map(str.title)
    data["Plant Stage"] = data["Plant Stage"].map(str.title)
    print("Standardized 'Plant Type' and 'Plant Stage' to Title Case.")
    
    # 3. Concatenate 'Plant Type' and 'Plant Stage' into a new 'Plant Type-Stage' column
    data['Plant Type-Stage'] = data['Plant Type'] + '-' + data['Plant Stage']
    print("Created 'Plant Type-Stage' by concatenating 'Plant Type' and 'Plant Stage'.")
    
    return data

# Function for converting numeric data to numeric type
def convert_to_numeric(data):
    """
    Convert specified columns to numeric type, coercing errors to NaN.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to numeric.
    """
    columns_to_numeric = [
        'Temperature Sensor (°C)', 'Humidity Sensor (%)', 'Light Intensity Sensor (lux)',
        'Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)',
        'Water Level Sensor (mm)'
    ]
    for col in columns_to_numeric:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        print(f"Converted {col} to numeric type, coercing errors to NaN.")
    return data

# Function to handle outliers
def handle_outliers(data):
    """
    Handle outliers in the dataset based on specific conditions, such as flipping negative temperature values,
    removing negative values for light intensity and EC, and removing extreme outliers using the IQR method.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame after handling outliers.
    """
    # 1. Flip negative temperature values to positive
    data['Temperature Sensor (°C)'] = data['Temperature Sensor (°C)'].abs()
    print("Flipped negative temperature values to positive.")

    # 2. Remove rows with negative Light Intensity values
    data = data[data['Light Intensity Sensor (lux)'] >= 0]
    print("Removed rows with negative Light Intensity values.")

    # 3. Remove negative EC values
    data = data[data['EC Sensor (dS/m)'] >= 0]
    print("Removed rows with negative EC values.")

    # 4. Identify and remove extreme EC outliers using the IQR method
    Q1 = data['EC Sensor (dS/m)'].quantile(0.25)
    Q3 = data['EC Sensor (dS/m)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data[(data['EC Sensor (dS/m)'] >= lower_bound) & (data['EC Sensor (dS/m)'] <= upper_bound)]
    print("Removed extreme EC outliers using the IQR method.")

    return data

# Main Preprocessing Function
def preprocess_data(df, target_type, config):
    """
    Preprocess the data by handling outliers, imputing missing values, scaling numeric features,
    and encoding categorical features for either temperature or plant_stage prediction.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_type (str): The target type to process ('temperature' or 'plant_stage').
        config (dict): Configuration dictionary containing feature columns and target variable.

    Returns:
        np.ndarray: Preprocessed feature matrix.
        pd.Series: Target variable.
    """
    # Standardize categories
    df = standardize_format(df)

    # Convert specified columns to numeric
    df = convert_to_numeric(df)

    # Handle outliers
    df = handle_outliers(df)

    # Select the correct target column based on target_type
    if target_type == "temperature":
        target_column = config["temperature_target_column"]
    elif target_type == "plant_stage":
        target_column = config["plant_stage_target_column"]
    else:
        raise ValueError("Invalid target_type. Must be 'temperature' or 'plant_stage'.")

    # Extract numeric and categorical features
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]

    # Define transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with the mean
        ('scaler', StandardScaler())                 # Scale numeric features
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the mode
        ('encoder', OneHotEncoder(handle_unknown='ignore'))    # Encode categorical features
    ])
    
    # Combine numeric and categorical preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Separate features (X) and target variable (y)
    X = df[numeric_features + categorical_features]
    y = df[target_column]

    # Apply preprocessing pipeline
    X_processed = preprocessor.fit_transform(X)

    # Save the preprocessor for future use
    joblib.dump(preprocessor, f"preprocessor_{target_type}.pkl")
    print(f"Preprocessing pipeline for {target_type} saved to preprocessor_{target_type}.pkl")

    return X_processed, y

if __name__ == "__main__":
    # Get target type from command-line argument
    target_type = sys.argv[1]  # Expects 'temperature' or 'plant_stage'

    # Load configuration and dataset
    with open("config.json", "r") as f:
        config = json.load(f)
    df = pd.read_csv("data.csv")

    # Preprocess the data based on the target type
    X, y = preprocess_data(df, target_type, config)

    # Display success message
    print(f"Data preprocessing complete for {target_type}. Preprocessed features and target variable are ready.")
