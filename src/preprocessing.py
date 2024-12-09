import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
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
def preprocess_data(df, config):
    """
    Preprocess the data by handling outliers, imputing missing values, scaling numeric features,
    and encoding categorical features for both temperature and plant_stage prediction.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        config (dict): Configuration dictionary containing feature columns and target variables.

    Returns:
        np.ndarray: Preprocessed feature matrix.
        pd.Series: Target variable for temperature prediction.
        pd.Series: Target variable for plant_stage categorization.
    """
    # Standardize categories
    df = standardize_format(df)

    # Convert specified columns to numeric
    df = convert_to_numeric(df)

    # Handle outliers
    df = handle_outliers(df)

    # Extract the target columns based on config
    y_temp = df[config["temperature_target_column"]]  # Target for temperature prediction
    y_cat = df[config["plant_stage_target_column"]]   # Target for plant stage categorization

    # Impute missing values in y_temp with the mean
    y_temp_imputer = SimpleImputer(strategy='mean')
    y_temp = pd.Series(y_temp_imputer.fit_transform(y_temp.values.reshape(-1, 1)).flatten(), index=y_temp.index)
    print("Imputed missing values in y_temp with the mean.")

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
    
    # Separate features (X)
    X = df[numeric_features + categorical_features]

    # Apply preprocessing pipeline
    X_processed = preprocessor.fit_transform(X)

    # Convert X_processed back to DataFrame for easier inspection
    X_final = pd.DataFrame(X_processed)

    # Optional: Add feature names back to the DataFrame (after OneHotEncoding)
    feature_names = numeric_features + preprocessor.transformers_[1][1]['encoder'].get_feature_names_out(categorical_features).tolist()
    X_final.columns = feature_names

    # Save the preprocessor for future use
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("Preprocessing pipeline saved to preprocessor.pkl")

    return X_final, X_processed, y_temp, y_cat

if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Load data
    df = pd.read_csv("data.csv")  # Adjust this if you're loading from SQLite

    # Preprocess the data
    X_final, X, y_temp, y_cat = preprocess_data(df, config)

    # Save the preprocessed data
    pd.DataFrame(X).to_pickle("X.pkl")  # Save feature matrix
    y_temp.to_pickle("y_temp.pkl")  # Save target for temperature prediction
    y_cat.to_pickle("y_cat.pkl")  # Save target for plant stage categorization

    # Print the final DataFrame for inspection
    # print("Final Preprocessed DataFrame:")
    # print(X_final.head())  # Display the first few rows of the processed features DataFrame

    print("Preprocessing complete. Files saved: X.pkl, y_temp.pkl, y_cat.pkl")
