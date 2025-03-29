
# Logistic Regression Model with AWS S3 Integration



                                                                                                                                
## 📌 Overview
- This project demonstrates how to train, save, and load a logistic regression model using AWS S3. 

- The dataset used for training is ObesityDataSet_raw_and_data_sinthetic.csv, and the model predicts whether an individual has a family history of being overweight based on their weight.

## Features
- Data Preprocessing: Uses LabelEncoder to encode categorical labels.

- Model Training: Trains a LogisticRegression model using scikit-learn.

- Model Persistence:

    - Saves the trained model locally using joblib.

    - Uploads the model to AWS S3.

    - Loads the model from AWS S3 for predictions.

- AWS S3 Integration:

    - Connects to an S3 bucket using boto3.

    - Handles credentials via a JSON file.

- Prediction: Uses the trained model to make predictions on new data.
## Installation

Ensure you have Python 3.x installed. Then, install the required dependencies:

```
 pip install pandas scikit-learn boto3 joblib
```
    
## Usage

-1. Set Up AWS Credentials

Create a JSON file named aws_creds.json in the root directory with the following format:

```
{
    "AWS_ACCESS_KEY_ID": "your-access-key",
    "AWS_SECRET_ACCESS_KEY": "your-secret-key"
}
```
-2. Train and Save Model to S3

Run the script to train the logistic regression model and upload it to AWS S3:

```
tr_model = train_model()
save_to_s3(tr_model)
```
-3. Load Model from S3 and Predict

To load the trained model from S3 and make predictions:
```
model = load_model_from_s3()
predictions = predict_with_model(model, x_test)
print(f"Model predictions: {predictions}")

```
## Note

- Ensure your AWS credentials are correct and that you have access to the specified S3 bucket.

- Modify S3_BUCKET_NAME and MODEL_FILE_KEY as per your AWS setup.

- The dataset should be available in the working directory before running the script.
