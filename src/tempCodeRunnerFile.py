import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path1, file_path2, columns):
    # Importing the both datasets of UNSW-NB15
    df1 = pd.read_csv(file_path1, names=columns, skiprows=1, low_memory=False)
    df2 = pd.read_csv(file_path2, names=columns, skiprows=1, low_memory=False)

    # Concatenating the two datasets into a single DataFrame
    df = pd.concat([df1, df2], ignore_index=True)

    # Displaying basic information about the DataFrame
    df.info()

    # Displaying the first few rows of the DataFrame
    print(df.head())

    # Checking for missing values in the DataFrame
    print(df.isnull().sum())

    # List all the columns in the DataFrame
    print(df.columns.tolist())

    # Displaying the unique values in the 'label' column
    print(df['label'].value_counts())

    return df

def preprocess_dataframe(df):
    # Dropping unnecessary columns from the DataFrame
    df.drop(columns=['srcip', 'sport', 'dstip', 'dsport', 'attack_cat'], inplace=True)

    # Columns with string values that need to be encoded
    cat_columns = ['proto', 'service', 'state']

    # Creating encoder object
    label_encoders = {}

    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Storing the encoder for later use

    # Confirming that all values are now numeric
    print(df.dtypes.value_counts())

    # Checking class balance
    print(df['label'].value_counts(normalize=True))

    # Check for missing values
    print("Missing values:", df.isnull().sum().sum())

    # Confirm dataset shape
    print("Shape of dataset:", df.shape)

    return df, label_encoders

def split_and_clean(df):
    # Splitting the dataset into features (X) and target variable (y)
    X = df.drop(columns=['label'])
    y = df['label']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Find non-numeric columns in X_train
    non_numeric_cols = X_train.select_dtypes(include='object').columns
    print("Non-numeric columns:\n", non_numeric_cols)

    # Converting 'ct_ftp_cmd' column to numeric, replacing spaces with NaN
    X_train['ct_ftp_cmd'] = pd.to_numeric(X_train['ct_ftp_cmd'].replace(' ', np.nan), errors='coerce')
    X_test['ct_ftp_cmd'] = pd.to_numeric(X_test['ct_ftp_cmd'].replace(' ', np.nan), errors='coerce')

    print(X_train['ct_ftp_cmd'].unique())

    # Step 2: Fill NaNs with median
    X_train['ct_ftp_cmd'].fillna(X_train['ct_ftp_cmd'].median(), inplace=True)
    X_test['ct_ftp_cmd'].fillna(X_test['ct_ftp_cmd'].median(), inplace=True)

    # Displaying the shapes of the training and testing sets
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)

    # Checking for NaN values in the training and testing sets
    print("NaNs in X_train:", np.isnan(X_train).sum().sum())
    print("NaNs in X_test:", np.isnan(X_test).sum().sum())

    # Fill all missing values with the median of each column
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_test.median(numeric_only=True))

    # Checking for NaN values again after filling
    print("✅ NaNs in X_train after clean-up:", X_train.isnull().sum().sum())
    print("✅ NaNs in X_test after clean-up:", X_test.isnull().sum().sum())

    return X_train, X_test, y_train, y_test