from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd

# Function to train models for temperature prediction
def train_temperature_models(X, y):
    """
    Trains two models for temperature prediction: Random Forest and Gradient Boosting Regression.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (pd.Series): Target variable (temperature).

    Returns:
        None
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    print(f"Random Forest Temperature Prediction MAE: {rf_mae}")
    joblib.dump(rf_model, "temperature_rf_model.pkl")

    # Model 2: Gradient Boosting Regressor
    print("Training Gradient Boosting Regressor...")
    gbr_model = GradientBoostingRegressor(random_state=42)
    gbr_model.fit(X_train, y_train)
    gbr_predictions = gbr_model.predict(X_test)
    gbr_mae = mean_absolute_error(y_test, gbr_predictions)
    print(f"Gradient Boosting Temperature Prediction MAE: {gbr_mae}")
    joblib.dump(gbr_model, "temperature_gbr_model.pkl")


# Function to train models for plant type-stage categorization
def train_categorization_models(X, y):
    """
    Trains two models for plant type-stage categorization: Random Forest and Logistic Regression.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (pd.Series): Target variable (plant type-stage category).

    Returns:
        None
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Random Forest Categorization Accuracy: {rf_accuracy}")
    joblib.dump(rf_model, "categorization_rf_model.pkl")

    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=500, random_state=42)  # Increase max_iter to ensure convergence
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"Logistic Regression Categorization Accuracy: {lr_accuracy}")
    joblib.dump(lr_model, "categorization_lr_model.pkl")

if __name__ == "__main__":
    # Load processed features and targets
    X = pd.read_pickle("X.pkl")  # Feature matrix
    y_temp = pd.read_pickle("y_temp.pkl")  # Target for temperature prediction
    y_cat = pd.read_pickle("y_cat.pkl")  # Target for plant categorization

    # Train models for temperature prediction
    print("Training models for Temperature Prediction...")
    train_temperature_models(X, y_temp)

    # Train models for plant type-stage categorization
    print("\nTraining models for Plant Type-Stage Categorization...")
    train_categorization_models(X, y_cat)
