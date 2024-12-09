# AIAP Batch 19 Technical Assessment

Create Machine Learning models to predict the temperature conditions within the farm's closed environment, ensuring optimal plant growth. 
Additionally, develop models to categorise the combined "Plant Type-Stage" based on sensor data, aiding in strategic planning and resource allocation.
By implementing these models, I will help AgroTech Innovations improve crop management, optimise resource usage, and increase yield predictability. 
These efforts will not only enhance current operations but also provide valuable insights for future agricultural innovations.

## Table of Contents

- [Full Name](#fullname)
- [Overview](#overview)
- [Installation](#installation)
- [Description of Pipeline](#description)
- [Overview of Key Findings from EDA](#keyfindings)
- [Describe How Features In the Dataset Are Processed](#featuresprocessing)
- [Explanation of Choice of Models] (#modelexplanation)
- [Other Considerations] (#otherconsiderations)


## Full Name
Full name (as in NRIC) and email address (stated in your application form).

Full Name: Tan Ping Min, Jasmine
Email Address: jasmine.tpm@gmail.com

## Overview
Overview of the submitted folder and the folder structure.

```
project-folder/
├── data/                     # Directory for storing datasets (e.g., raw and processed data).
├── src/                      # Directory containing core Python scripts (4 files).
│   ├── data_ingestion.py     # Script for ingesting and loading data into the pipeline.
│   ├── preprocessing.py      # Script for cleaning and preprocessing the data.
│   ├── model_training.py     # Script for training and evaluating models.
│   └── config.json           # Configuration file for the project
├── .gitignore                # Specifies files and directories to be excluded from Git tracking.
├── eda.ipynb                 # Jupyter Notebook for Exploratory Data Analysis.
├── README.md                 # Documentation providing an overview and instructions for the project.
├── requirements.txt          # List of required Python packages for the project.
├── run.sh                    # Bash script to execute the pipeline and orchestrate tasks.
```

## Installation
Instructions for executing the pipeline and modifying any parameters.

1. Clone the repository:
   ```bash
   git clone https://github.com/jasminetpm/aiap19-tan-ping-min-jasmine-108J.git

2. Setup Environment
Run in Console:
```
python -m venv venv
source .venv/bin/activate
pip install -r requirements.txt
```

All software, packages, and dependencies that need to be installed beforehand are listed in requirements.txt

3. Execute the ML Pipeline
To execute the bash script, run in Console:
```
**bash run.sh**
```
4. Check Outputs

- Models saved as .pkl files.
- Preprocessor saved as preprocessor.pkl.


## Description of Pipeline
Description of logical steps/flow of the pipeline. 

1. Data Ingestion (src/data_ingestion.py)

- Reads raw data from the specified path in config.json (reads data from agri.db in the data folder).
- Validates the data for consistency, handling missing or corrupted records.
- Saves the ingested data in a standardized format (csv file) for preprocessing.

2. Exploratory Data Analysis (EDA) (eda.ipynb)

- Provides insights into the data using visualizations and statistical summaries.
- Identifies key patterns, correlations, and distributions.
- Helps refine preprocessing or feature engineering steps (i.e. to know which parts of the data to clean and how to clean).

3. Data Preprocessing (src/preprocess.py)

Loads the ingested data.
- Cleans the data (e.g., handling missing values, outliers, and converts all data to correct type). More insights into data cleaning can be found in the EDA.
Encodes categorical variables and scales numerical features.
We encode categorical variables because machine learning models typically operate on numerical data. Categorical variables, which contain text or labels, need to be transformed into numerical format for the model to process them.
We scale numerical variables to ensure that all numerical features have comparable ranges, making models perform better and converge faster. For instance, models like logistic regression that I use in my pipeline are sensitive to feature scales. Large feature values can dominate smaller ones, leading to biased model performance.

- Splits the data into feature matrix (X) and target variables (y).
- Stores the processed data in pickle files (X.pkl, y_temp.pkl and y_cat.pkl).

4. Model Training and Prediction (src/model_training.py)

- Loads the preprocessed feature matrix (X.pkl) and target variables (y_temp.pkl and y_cat.pkl).
- Trains predictive models (i.e., Random Forest and Gradient Boosting Regression for *Temperature Prediction*; and Random Forest Classifier and Logistic Regression for *Plant Type-Stage Prediction*- ).
Evaluates models using metrics such as mean absolute error (MAE), accuracy, precision, recall, F1 score and confusion matrix.
Saves trained models as serialized files (e.g., model.pkl) for later use.


5. Configuration and Parameters (config.json)

Central repository for configuration settings such as:
- File paths for data and models.
- Target Features
- Numeric and Categorical Features

6. Bash script that automates the sequential execution of the pipeline:
Runs data_ingestion.py.
Runs preprocess.py.
Executes model_training.py.
Ensures the pipeline runs smoothly from data ingestion to model training with a single command.

7. Documentation and Requirements
- README.md: Provides an overview of the project, setup instructions, and pipeline usage.
- requirements.txt: Lists all Python dependencies required for the project.
- .gitignore: Specifies files and folders to exclude from version control (e.g., intermediate files, logs, and serialized models).


## Overview of Key Findings from EDA

1. Data Quality Insights
- Missing Data: Certain features had missing values (e.g. Humidity data had a high quanityty of missing data), indicating the need for imputation during preprocessing.
- Outliers: Detected outliers in some numerical columns through boxplots and statistical analysis. Outlier treatment (e.g. removal) was incorporated into the ML Pipeline to avoid skewing model.performance.

2. Feature Relationships
- Correlations were observed between features, such as Nutrients N, P, K Levels, Light Intensity and CO2 Levels.

3. Numerical Features
- Numerical features had varying ranges (e.g., Light Intensity (lux) ranged from -800 to 800, while Electrical Conductivity ranged from ~ 0.0 to 3.5). Scaling was applied in the ML Pipeline to ensure all numerical features were on the same scale for optimal model performance.

4. Categorical Features
- Key categorical features, such as plant type and stage, were non-numeric. One-hot encoding was used to encode these low-cardinality variables.

5. Data Distributions and Patterns
- Clustering patterns were identified in the 3D plots and scatterplots, suggesting possible subgroups in the data, especially based on Plant Type and Stage.

### Pipeline Choices Based on EDA Findings

1. Preprocessing:
- Missing value imputation.
- Cleaning Data
- Handling of Outliers
- Scaling of numerical features.
- Encoding of categorical features.

2. Model Training:
- I should have used correlation insights to prioritize important features (eg Light Intensity and CO2 Levels to predict Plant Type and Stage).

## Describe How Features In the Dataset Are Processed (summarised in a table)

| **Feature**                     | **Type**         | **Issues Identified**                | **Processing Steps**                                                                                                                                 |
|----------------------------------|------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **System Location Code**         | Categorical      | High Cardinality                     | - Encode using target encoding to reduce dimensionality.                                                                                             |
| **Previous Cycle Plant Type**    | Categorical      | Missing Values                       | -  One-hot encoding for categorical representation.                                                        |
| **Plant Type**                   | Categorical      | Weird Formatting, Missing Values                       | - Standardized 'Plant Type' and 'Plant Stage' to Title Case. <br> -  Encode using one-hot encoding                                 |
| **Plant Stage**                  | Categorical      | Weird Formatting, Low Cardinality                      | - Standardized 'Plant Type' and 'Plant Stage' to Title Case. <br> - One-hot encode to create binary features for each category.                                                                                        |
| **Temperature Sensor (°C)**      | Numerical        | Missing Values, Outliers             | - Impute missing values with mean. <br> - Handle outliers using absolute. <br> - Scale using StandardScaler.         |
| **Humidity Sensor (%)**          | Numerical        | Missing Values                       | - Impute missing values using mean. <br> - Scale for consistency with other numerical features.                                            |
| **Light Intensity Sensor (lux)** | Numerical        | Outliers                             | - Handle outliers using clipping. Remove values less than zero. <br> - Scale for consistency.                                                                   |
| **CO2 Sensor (ppm)**             | Numerical        | Outliers          | - Scale using StandardScaler.            |
| **EC Sensor (dS/m)**             | Numerical        | Negative Values     | - Remove negative values <br> - Use the Interquartile Range (IQR) method to identify and remove extreme outliers: Scale using StandardScaler.                        |
| **O2 Sensor (ppm)**              | Numerical        | None                     | - Impute missing values using mean. <br> - Scale for consistency with other numerical features.                                                      |
| **Nutrient N Sensor (ppm)**      | Numerical        | Missing Values, Outliers             | - Impute missing values with mean.  <br> -  Stripped ' ppm' and converted Nutrient N, P and K Sensor (ppm) to numeric.                    |
| **Nutrient P Sensor (ppm)**      | Numerical        | Missing Values, Outliers             | - Impute missing values using mean.  <br> -  Stripped ' ppm' and converted Nutrient N, P and K Sensor (ppm) to numeric.                                  |
| **Nutrient K Sensor (ppm)**      | Numerical        | Missing Values, Outliers             | - Impute missing values using mean.   <br> -  Stripped ' ppm' and converted Nutrient N, P and K Sensor (ppm) to numeric.                                      |
| **pH Sensor**                    | Numerical        | None    | - Impute missing values using mean. <br> - Standardize values to a consistent scale (e.g., 0-14).                                                    |
| **Water Level Sensor (mm)**      | Numerical        | Missing Values,              | - Impute missing values with mean. |


## Explanation of Choice of Models for Each Machine Learning Task

### Task: Temperature Prediction

#### Model: Random Forest Regressor
Reasons:
- Handles non-linear relationships well.
- Robust to outliers and noise in the data.
- Automatically captures feature importance and interactions.

#### Model: Gradient Boosting Regressor
Reasons:
- Performs well with moderately sized datasets and complex patterns.
- Optimizes errors sequentially, reducing bias.
- Highly flexible for fine-tuning.

#### Key Considerations:
Both models are ensemble methods capable of capturing complex patterns in the data.
Random Forest focuses on reducing variance, making it suitable for datasets with potential overfitting issues.
Gradient Boosting sequentially reduces bias and achieves better performance on datasets with subtle patterns.

### Task: Plant Type-Stage Prediction

#### Model: Random Forest Classifier
Reasons:
- Effective for multi-class classification tasks.
- Handles high-dimensional and categorical features well/ Handles non-linear relationships well.
- Offers feature importance insights.
- Robust against overfitting due to ensemble averaging.
- Works well with high-dimensional data and categorical variables.


#### Model: Logistic Regression
Reasons:
- Suitable baseline model for classification.
- Interpretable coefficients to understand feature contributions.
- Performs well on linearly separable data.

#### Key Considerations:
Random Forest is chosen for its robustness and ability to handle non-linear relationships in the data.
Logistic Regression provides a simple, interpretable model and serves as a benchmark for performance comparison.


## Evaluation of the Models Developed. 
Any metrics used in the evaluation should also be explained.

### Task: Temperature Prediction

#### Model: Random Forest Regressor and Gradient Boosting Regressor


##### **Random Forest Results:**
**Training MAE: 2.43e-05**

**Test MAE: 6.17e-05**

##### **Gradient Boosting Results:**
**Training MAE: 0.00891**

**Test MAE: 0.00915**

**Analysis:**

##### **Random Forest Model:**

**Low Training MAE (2.43e-05)**: This value is extremely low, suggesting that the model is fitting the training data very well, with very small errors.
**Higher Test MAE (6.17e-05)**: While still small, the test MAE is slightly higher than the training MAE, indicating that the model might not generalize perfectly to unseen data. However, this is not a major concern because the test MAE is still very small, suggesting that the Random Forest model is still making accurate predictions on the test set.
Interpretation: The Random Forest model is performing well, but there might be a slight overfitting to the training data. Overfitting is common when the model learns very specific patterns in the training data that do not generalize perfectly. However, the error is still very small and acceptable.

##### **Gradient Boosting Model:**

**Training MAE (0.00891)**: This is slightly higher than the Random Forest model's training MAE, indicating that the Gradient Boosting model isn't fitting the training data as perfectly.
**Test MAE (0.00915)**: The difference between training and test MAE is very small, suggesting that Gradient Boosting is generalizing better than Random Forest. It is likely avoiding overfitting, as the error on both the training and test sets is very similar.
Interpretation: The Gradient Boosting model has a slightly higher error on the training set, but its test error is almost the same, showing that it might be more stable and better at generalizing to new data. This is a positive sign of robustness and the model's ability to handle unseen data.

**Conclusion/ Pipeline Choice:**
Both models are performing well in terms of temperature prediction, with Random Forest being more accurate on the training set but showing a slight increase in test error. This could indicate some level of overfitting.
Gradient Boosting, while slightly less accurate on the training set, has similar performance on both training and test sets, indicating better generalization.
Since we are aiming for a balance between accuracy and generalization to predict the temperature conditions within the farm's closed environment, ensuring optimal plant growth, Gradient Boosting might be the better choice. However, if minimizing the training error is critical, then Random Forest might be preferred, as long as we can manage the slight overfitting.

### Task: Plant Type-Stage Prediction

#### Model: Random Forest Classifier and Logistic Regression

##### **Random Forest Classifier**

**Random Forest Categorization Accuracy: 1.0**

**Random Forest Precision: 1.0**

**Random Forest Recall: 1.0**

**Random Forest F1 Score: 1.0**

**Random Forest Confusion Matrix:**

```
[[1632    0    0    0    0    0    0    0    0    0    0    0]
 [   0 1744    0    0    0    0    0    0    0    0    0    0]
 [   0    0 1722    0    0    0    0    0    0    0    0    0]
 [   0    0    0 1792    0    0    0    0    0    0    0    0]
 [   0    0    0    0 1640    0    0    0    0    0    0    0]
 [   0    0    0    0    0 1636    0    0    0    0    0    0]
 [   0    0    0    0    0    0 1723    0    0    0    0    0]
 [   0    0    0    0    0    0    0 1781    0    0    0    0]
 [   0    0    0    0    0    0    0    0 1668    0    0    0]
 [   0    0    0    0    0    0    0    0    0 1669    0    0]
 [   0    0    0    0    0    0    0    0    0    0 1697    0]
 [   0    0    0    0    0    0    0    0    0    0    0 1691]]
 ```
**Random Forest Train Accuracy: 1.0**


##### **Logistic Regression**

**Logistic Regression Categorization Accuracy: 1.0**

**Logistic Regression Precision: 1.0**

**Logistic Regression Recall: 1.0**

**Logistic Regression F1 Score: 1.0**

**Logistic Regression Confusion Matrix:**
```
[[1632    0    0    0    0    0    0    0    0    0    0    0]
 [   0 1744    0    0    0    0    0    0    0    0    0    0]
 [   0    0 1722    0    0    0    0    0    0    0    0    0]
 [   0    0    0 1792    0    0    0    0    0    0    0    0]
 [   0    0    0    0 1640    0    0    0    0    0    0    0]
 [   0    0    0    0    0 1636    0    0    0    0    0    0]
 [   0    0    0    0    0    0 1723    0    0    0    0    0]
 [   0    0    0    0    0    0    0 1781    0    0    0    0]
 [   0    0    0    0    0    0    0    0 1668    0    0    0]
 [   0    0    0    0    0    0    0    0    0 1669    0    0]
 [   0    0    0    0    0    0    0    0    0    0 1697    0]
 [   0    0    0    0    0    0    0    0    0    0    0 1691]]
 ```
**Logistic Regression Train Accuracy: 1.0**

There seems to be something wrong with the model as there is 100% accuracy. I will have to figure this out.

I will still explain the metrics used to evaluate the model:

- Accuracy: Proportion of correctly predicted instances out of the total predictions.
A value of 1.0 indicates the model predicted all instances correctly.

- Precision: Proportion of true positives (correctly classified instances) out of all predicted positives for each class.

- Weighted average is used due to multiple classes.
A value of 1.0 signifies no false positives.

- Recall (Sensitivity): Proportion of true positives out of all actual positives for each class.
A value of 1.0 indicates no false negatives.

- F1 Score: Harmonic mean of precision and recall, balancing the two metrics.
A value of 1.0 indicates perfect precision and recall.

- Confusion Matrix: Summarizes the performance by showing true positives, false positives, and false negatives for each class.
A diagonal matrix (non-zero entries only on the diagonal) signifies perfect classification.

**Conclusion/ Pipeline Choice:**
I would prefer Random Forest for its robustness and non-linear feature handling. It would likely generalize better on unseen data in slightly noisier scenarios.


## Other Considerations
- In the future, I would probably remove more features from the dataset that are lowly-correlated to the target feature *Plant Type-Stage* to prevent overfitting and increase predictive performance.
- I will also apply different encoding strategies for Plant Type and Plant Stage to better capture their unique characteristics. Since Plant Type represents distinct, unrelated categories, a one-hot encoding approach is appropriate to treat each type independently without implying any ordinal relationship. On the other hand, Plant Stage follows a sequential progression (e.g. Seedling progresses to Vegetative progresses to Maturity), making ordinal encoding or a similar method more suitable to preserve the inherent order and capture meaningful stage transitions.