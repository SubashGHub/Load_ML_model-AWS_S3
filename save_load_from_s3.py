from io import BytesIO
import boto3
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
data.head(2)

label_encode = LabelEncoder()
y_encode = label_encode.fit_transform(data['family_history_with_overweight'])

x = data[['Weight']]
y = y_encode
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


def get_aws_creds(filepath='aws_creds.json'):
    try:
        with open(filepath, "r") as f:
            creds = json.load(f)
            return creds['AWS_ACCESS_KEY_ID'], creds['AWS_SECRET_ACCESS_KEY']
    except FileNotFoundError as e:
        print(f'File {filepath} not found. {e}')
    except KeyError:
        print("Error: Invalid JSON structure. Expected keys: 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'.")


# Configuration - replace these with your values
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = get_aws_creds()
S3_BUCKET_NAME = 'your bucket name'
MODEL_FILE_KEY = 'model/logi_reg.joblib'


def conn_s3():
    # Initialize S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    print('conn done.')
    return s3


def load_model_from_s3():
    """
    Loads a trained ML model from AWS S3 bucket
    Returns:
        model: The loaded ML model ready for predictions
    """
    conn = conn_s3()

    # Get the model file from S3
    try:
        response = conn.get_object(Bucket=S3_BUCKET_NAME, Key=MODEL_FILE_KEY)
        model_str = response['Body'].read()

        # Load the model from bytes
        model = joblib.load(BytesIO(model_str))
        print("Model loaded successfully from S3")
        return model
    except Exception as e:
        print(f"Error loading model from S3: {str(e)}")
        raise


def save_to_s3(trained_model):
    # Create in-memory file object
    buffer = BytesIO()
    joblib.dump(trained_model, buffer)
    buffer.seek(0)  # rewind pointer to start of buffer

    # Upload to S3
    conn = conn_s3()
    conn.upload_fileobj(buffer, S3_BUCKET_NAME, MODEL_FILE_KEY)
    print('Model uploaded successfully to S3 bucket.')


def train_model():
    reg = LogisticRegression(random_state=10)
    # path to the model fite
    model_file = 'logi_reg.joblib'
    model_trained = reg.fit(x_train, y_train)

    # save the model to the current directory
    joblib.dump(model_trained, model_file)
    print('model trained done.')
    return model_trained


def predict_with_model(model, input_data):
    """
    Makes predictions using the loaded model
    Args:
        model: The loaded ML model
        input_data: New data for prediction (as DataFrame or array-like)
    Returns:
        predictions: Model predictions
    """
    try:
        predictions = model.predict(input_data)
        print('Prediction done.')
        return predictions
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        raise


# model = load_model_from_s3()
tr_model = train_model()
save_to_s3(tr_model)
# predictions = predict_with_model(model, x_test)
# print(f"Model predictions: {predictions}")
