import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df, target_column, categorical_columns, numerical_columns):
    """
    Prepares dataset for training: encodes categorical variables, scales numerical features.
    """
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Create a preprocessing and modelling pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Preprocess the data
    X_train = model_pipeline.fit_transform(X_train)
    X_test = model_pipeline.transform(X_test)

    return X_train, X_test, y_train, y_test
