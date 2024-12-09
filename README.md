# AIAP Batch 19 Technical Assessment

Create Machine Learning models to predict the temperature conditions within the farm's closed environment, ensuring optimal plant growth. 
Additionally, develop models to categorise the combined "Plant Type-Stage" based on sensor data, aiding in strategic planning and resource allocation.
By implementing these models, I will help AgroTech Innovations improve crop management, optimise resource usage, and increase yield predictability. 
These efforts will not only enhance current operations but also provide valuable insights for future agricultural innovations.

## Table of Contents

- [Submission](#submission)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Submission
Full Name: Tan Ping Min, Jasmine
Email Address: jasmine.tpm@gmail.com

## Installation


### Prerequisites
All software, packages, and dependencies that need to be installed beforehand are listed in requirements.txt

Please install them by running:
pip install -r requirements.txt

### How to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git

2. Setup Environment
Run in Console:

python -m venv venv
source .venv/bin/activate
pip install -r requirements.txt

3. Execute the ML Pipeline
Run in Console:

bash run.sh

4. Check Outputs

- Models saved as .pkl files (temperature_model.pkl, categorization_model.pkl).
- Preprocessor saved as preprocessor.pkl.

## Overview of Key Findings from EDA 
- Stripped ' ppm' and converted Nutrient N, P and K Sensor (ppm) to numeric.
- Standardized 'Plant Type' and 'Plant Stage' to Title Case.
- Flipped negative temperature values to positive.
- Removed rows with negative Light Intensity values.
- Removed rows with negative EC values.
- Removed extreme EC outliers using the IQR method.

## g. Explanation of your choice of models for each machine learning task.

## h. Evaluation of the models developed. Any metrics used in the evaluation should also be explained.

Random Forest Temperature Prediction MAE (Train): 2.426203792777704e-05
Random Forest Temperature Prediction MAE (Test): 6.171798399206995e-05

Gradient Boosting Temperature Prediction MAE (Train): 0.00890701037514177
Gradient Boosting Temperature Prediction MAE (Test): 0.009150800025774671

The results indicate the performance of the models on both the training and test datasets. Let's break it down:

Random Forest Results:
Training MAE: 2.43e-05
Test MAE: 6.17e-05

Gradient Boosting Results:
Training MAE: 0.00891
Test MAE: 0.00915

Analysis:

Random Forest Model:

Low Training MAE (2.43e-05): This value is extremely low, suggesting that the model is fitting the training data very well, with very small errors.
Higher Test MAE (6.17e-05): While still small, the test MAE is slightly higher than the training MAE, indicating that the model might not generalize perfectly to unseen data. However, this is not a major concern because the test MAE is still very small, suggesting that the Random Forest model is still making accurate predictions on the test set.
Interpretation: The Random Forest model is performing well, but there might be a slight overfitting to the training data. Overfitting is common when the model learns very specific patterns in the training data that do not generalize perfectly. However, the error is still very small and acceptable.

Gradient Boosting Model:

Training MAE (0.00891): This is slightly higher than the Random Forest model's training MAE, indicating that the Gradient Boosting model isn't fitting the training data as perfectly.
Test MAE (0.00915): The difference between training and test MAE is very small, suggesting that Gradient Boosting is generalizing better than Random Forest. It is likely avoiding overfitting, as the error on both the training and test sets is very similar.
Interpretation: The Gradient Boosting model has a slightly higher error on the training set, but its test error is almost the same, showing that it might be more stable and better at generalizing to new data. This is a positive sign of robustness and the model's ability to handle unseen data.

Conclusion:
Both models are performing well in terms of temperature prediction, with Random Forest being more accurate on the training set but showing a slight increase in test error. This could indicate some level of overfitting.
Gradient Boosting, while slightly less accurate on the training set, has similar performance on both training and test sets, indicating better generalization.
Since we are aiming for a balance between accuracy and generalization to predict the temperature conditions within the farm's closed environment, ensuring optimal plant growth, Gradient Boosting might be the better choice. However, if minimizing the training error is critical, then Random Forest might be preferred, as long as we can manage the slight overfitting.