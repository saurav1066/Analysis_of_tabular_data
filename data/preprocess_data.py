import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

#Function to load heart disease data
def load_heart_disease_data():
    # Load the heart disease data
    data = pd.read_csv('data/heart.csv')
    return data

#Function to load breast cancer data
def load_breast_cancer_data():
    # Load the breast cancer data
    data = pd.read_csv('data/breast_cancer.csv')
    return data

#Function to preprocess heart disease data
def preprocess_heart_disease_data(data, columns):
    # Drop rows with missing values
    data = data.dropna()
    # Drop duplicate rows
    data = data.drop_duplicates()

#function that takes a dataframe and checks for categorical columns and then use ordinal encoder on the categoriccl columns
def encode_categorical_columns(data):
    # Identify the categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # If no categorical columns
    if len(categorical_columns) == 0:
        return data

    # Create a label (ordinal) encoder
    encoder = OrdinalEncoder()

    # Encode the categorical columns
    data[categorical_columns] = encoder.fit_transform(data[categorical_columns])

    return data

#Function to convert the num column in the heart disease data to binary
def convert_num_column(data):
    # Convert the num column to binary
    data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)
    return data

# Function to split the data into trsin and test splits and convert them to csv
def split_data(data, filename):
    # Split the data into train and test sets
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)

    # Save the data to CSV
    train.to_csv(f'data/{filename}_train.csv', index=False)
    test.to_csv(f'data/{filename}_test.csv', index=False)


#Function to preprocess breast cancer data
def preprocess_breast_cancer_data(data, columns):
    # Drop rows with missing values
    data = data.dropna()
    # Drop duplicate rows
    data = data.drop_duplicates()

    # Encode the categorical columns
    data = encode_categorical_columns(data)

    return data

#Function to convert the diagnosis column in the breast cancer data to binary
def convert_diagnosis_column(data):
    # Convert the diagnosis column to binary
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    return data

