import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the trained model and encoders/imputer
try:
    model = joblib.load('gradient_boosting_regressor_tuned.joblib')
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    label_encoder_kitchenqual = joblib.load('label_encoder_kitchenqual.joblib')
except FileNotFoundError:
    st.error("Model or encoder files not found. Please make sure 'gradient_boosting_regressor_tuned.joblib', 'onehot_encoder.joblib', and 'label_encoder_kitchenqual.joblib' are in the same directory.")
    st.stop()

# Define the columns to drop and columns for encoding
columns_to_drop = ['Id', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold']
columns_to_encode = ['SaleType', 'SaleCondition']

# Streamlit app title
st.title("House Price Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.dataframe(df_input.head())

        # Preprocessing steps
        # Drop unnecessary columns
        df_processed = df_input.drop(columns=columns_to_drop, errors='ignore')

        # Apply label encoding to KitchenQual
        if 'KitchenQual' in df_processed.columns:
            df_processed['KitchenQual'] = df_processed['KitchenQual'].map(label_encoder_kitchenqual)

        # Apply one-hot encoding
        for col in columns_to_encode:
            if col in df_processed.columns:
                # Handle potential new categories during prediction
                # For simplicity, we will apply the transform directly.
                # In a production setting, you might need to handle unknown categories.
                try:
                    encoded_data = onehot_encoder.transform(df_processed[[col]])
                    encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out([col]))
                    df_processed = df_processed.drop(columns=[col])
                    df_processed = pd.concat([df_processed, encoded_df], axis=1)
                except ValueError as e:
                    st.warning(f"Could not one-hot encode column '{col}'. It might contain categories not seen during training. Error: {e}")
                    # Drop the column if encoding fails to avoid errors
                    df_processed = df_processed.drop(columns=[col])


        # Impute missing values using the pre-trained imputer
        # Identify columns with missing values in the processed data
        cols_with_missing_processed = df_processed.columns[df_processed.isnull().any()].tolist()
        if cols_with_missing_processed:
             # Create a new imputer instance for the specific columns to fit and transform
             # This assumes the training data and prediction data have similar distributions
             # For robust deployment, you might fit the imputer on training data and just transform here
             imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
             df_processed[cols_with_missing_processed] = imputer.fit_transform(df_processed[cols_with_missing_processed])

        st.write("Processed Data (first 5 rows after cleaning and encoding):")
        st.dataframe(df_processed.head())


        # Ensure columns match the training data columns
        # This is a crucial step to avoid errors during prediction
        # Get the list of columns the model was trained on (assuming X from previous steps is available or can be loaded)
        # For simplicity, we will assume the processed dataframe has the necessary columns after the above steps
        # In a real scenario, you would load or recreate the training columns list
        # trained_columns = joblib.load('trained_columns.joblib') # Example if you saved column names

        # Let's use the columns from the current processed dataframe for demonstration
        # In a real application, ensure these match the training columns
        # If the order or presence of columns differs, prediction will fail.
        # A robust method involves aligning columns using reindex.

        # Drop the 'SalePrice' column if it exists in the prediction data
        if 'SalePrice' in df_processed.columns:
            X_predict = df_processed.drop(columns=['SalePrice'])
        else:
            X_predict = df_processed.copy()

        # Select a row for prediction
        row_index = st.number_input("Enter the row index to predict (0-based)", min_value=0, max_value=len(X_predict)-1, value=0)

        if st.button("Predict Price"):
            # Get the selected row
            selected_row = X_predict.iloc[[row_index]]

            # Make prediction
            prediction = model.predict(selected_row)

            st.subheader(f"Predicted House Price for Row {row_index}:")
            st.write(f"${prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
