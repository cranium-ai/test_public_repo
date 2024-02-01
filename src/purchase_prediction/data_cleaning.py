import boto3
import pandas as pd

# S3 bucket details
bucket_name = 'data'
file_name = 'super_important_purchase_data.csv'

# Create a session using your AWS credentials
session = boto3.Session(
    aws_access_key_id='your_aws_access_key_id',
    aws_secret_access_key='your_aws_secret_access_key'
)

# Create a client for accessing the S3 bucket
s3_client = session.client('s3')

# Read the file from S3 bucket
response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
data = response['Body'].read().decode('utf-8')

# Create a pandas DataFrame from the data
data = pd.read_csv(pd.compat.StringIO(data))

# Basic cleaning steps
# Remove missing values
data.dropna(inplace=True)

# Convert age to integer
data['age'] = data['age'].astype(int)

# Convert time_on_page to seconds
data['time_on_page'] = pd.to_timedelta(data['time_on_page']).dt.total_seconds()

# Save the cleaned data to a new CSV
cleaned_file_name = "cleaned_purchase_data.csv"
data.to_csv(cleaned_file_name, index=False)
