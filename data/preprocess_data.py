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

