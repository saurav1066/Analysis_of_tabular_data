# preprocess_data.py
import pandas as pd
from load_data import load_data

def preprocess_data():
    data = load_data()
    # Replace '?' with NaN and convert data types
    data.replace('?', pd.NA, inplace=True)
    data = data.convert_dtypes()
    data['Bare Nuclei'] = pd.to_numeric(data['Bare Nuclei'], errors='coerce')
    data['Bare Nuclei'].fillna(data['Bare Nuclei'].mode()[0], inplace=True)
    return data
