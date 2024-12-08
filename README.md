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
source venv/bin/activate
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