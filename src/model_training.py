from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    
    # Predictions on training and test sets
    rf_train_predictions = rf_model.predict(X_train)
    rf_test_predictions = rf_model.predict(X_test)
    
    # Calculate MAE for both training and test sets
    rf_train_mae = mean_absolute_error(y_train, rf_train_predictions)
    rf_test_mae = mean_absolute_error(y_test, rf_test_predictions)
    
    print(f"Random Forest Temperature Prediction MAE (Train): {rf_train_mae}")
    print(f"Random Forest Temperature Prediction MAE (Test): {rf_test_mae}")
    
    joblib.dump(rf_model, "temperature_rf_model.pkl")

    # Model 2: Gradient Boosting Regressor
    print("Training Gradient Boosting Regressor...")
    gbr_model = GradientBoostingRegressor(random_state=42)
    gbr_model.fit(X_train, y_train)
    
    # Predictions on training and test sets
    gbr_train_predictions = gbr_model.predict(X_train)
    gbr_test_predictions = gbr_model.predict(X_test)
    
    # Calculate MAE for both training and test sets
    gbr_train_mae = mean_absolute_error(y_train, gbr_train_predictions)
    gbr_test_mae = mean_absolute_error(y_test, gbr_test_predictions)
    
    print(f"Gradient Boosting Temperature Prediction MAE (Train): {gbr_train_mae}")
    print(f"Gradient Boosting Temperature Prediction MAE (Test): {gbr_test_mae}")
    
    joblib.dump(gbr_model, "temperature_gbr_model.pkl")


def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plots a confusion matrix with proper class labels on axes.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Adjust layout and save
    plt.tight_layout()  # Automatically adjust spacing
    plt.subplots_adjust(left=0.2, bottom=0.3)  # Add padding for labels
    plt.savefig("confusion_matrix.png", bbox_inches="tight")  # Save as image
    plt.show()


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Model 1: Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    # Random Forest Accuracy
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Random Forest Categorization Accuracy: {rf_accuracy}")
    
    # Additional evaluation metrics for Random Forest
    rf_precision = precision_score(y_test, rf_predictions, average='weighted')
    rf_recall = recall_score(y_test, rf_predictions, average='weighted')
    rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
    print(f"Random Forest Precision: {rf_precision}")
    print(f"Random Forest Recall: {rf_recall}")
    print(f"Random Forest F1 Score: {rf_f1}")
    
    # Confusion Matrix for Random Forest
    class_names = y_cat.unique().tolist()  # Automatically infer class names from your target variable
    rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
    print(f"Random Forest Confusion Matrix:\n{rf_conf_matrix}")
    plot_confusion_matrix(rf_conf_matrix, class_names)
    
    joblib.dump(rf_model, "categorization_rf_model.pkl")

    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=500, random_state=42)  # Increase max_iter to ensure convergence
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    
    # Logistic Regression Accuracy
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"Logistic Regression Categorization Accuracy: {lr_accuracy}")
    
    # Additional evaluation metrics for Logistic Regression
    lr_precision = precision_score(y_test, lr_predictions, average='weighted')
    lr_recall = recall_score(y_test, lr_predictions, average='weighted')
    lr_f1 = f1_score(y_test, lr_predictions, average='weighted')
    print(f"Logistic Regression Precision: {lr_precision}")
    print(f"Logistic Regression Recall: {lr_recall}")
    print(f"Logistic Regression F1 Score: {lr_f1}")
    
    # Confusion Matrix for Logistic Regression
    lr_conf_matrix = confusion_matrix(y_test, lr_predictions)
    print(f"Logistic Regression Confusion Matrix:\n{lr_conf_matrix}")
    plot_confusion_matrix(rf_conf_matrix, class_names)

    
    joblib.dump(lr_model, "categorization_lr_model.pkl")

    # Optionally: Check for overfitting by comparing training and test accuracy
    rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
    print(f"Random Forest Train Accuracy: {rf_train_accuracy}")
    lr_train_accuracy = accuracy_score(y_train, lr_model.predict(X_train))
    print(f"Logistic Regression Train Accuracy: {lr_train_accuracy}")

    if rf_train_accuracy > rf_accuracy:
        print("Random Forest model may be overfitting (train accuracy > test accuracy).")
    if lr_train_accuracy > lr_accuracy:
        print("Logistic Regression model may be overfitting (train accuracy > test accuracy).")



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
