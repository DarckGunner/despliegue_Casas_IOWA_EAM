import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import traceback # Import traceback to get detailed error information

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
            # Handle potential NaN values in KitchenQual before mapping
            df_processed['KitchenQual'] = df_processed['KitchenQual'].map(label_encoder_kitchenqual).fillna(-1) # Use -1 or another indicator for unknown
        else:
             st.warning("KitchenQual column not found in the uploaded file.")


        # Apply one-hot encoding
        encoded_dfs = []
        for i, col in enumerate(columns_to_encode):
            if col in df_processed.columns:
                try:
                    # Use the loaded one-hot encoder's categories for this column
                    trained_categories = onehot_encoder.categories_[i]

                    # Convert the column to categorical type with known categories
                    df_processed[col] = pd.Categorical(df_processed[col], categories=trained_categories)

                    # Apply one-hot encoding
                    encoded_data = onehot_encoder.transform(df_processed[[col]])
                    encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out([col]))
                    encoded_dfs.append(encoded_df)

                except ValueError as e:
                    st.warning(f"Could not one-hot encode column '{col}'. It might contain categories not seen during training or the encoder categories do not match. Error: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during one-hot encoding of column '{col}': {e}")
                    st.error(traceback.format_exc()) # Print traceback for debugging
            else:
                st.warning(f"Column '{col}' not found in the uploaded file for one-hot encoding.")

        # Drop original columns before concatenating encoded ones
        df_processed = df_processed.drop(columns=columns_to_encode, errors='ignore')

        # Concatenate all encoded DataFrames
        for encoded_df in encoded_dfs:
             df_processed = pd.concat([df_processed, encoded_df], axis=1)


        # Impute missing values using the pre-trained imputer
        # Identify columns with missing values in the processed data
        cols_with_missing_processed = df_processed.columns[df_processed.isnull().any()].tolist()
        if cols_with_missing_processed:
             st.write(f"Imputing missing values in columns: {cols_with_missing_processed}")
             try:
                 # Use the loaded imputer directly to transform the data
                 # This assumes the imputer was fitted on the training data and the columns match
                 # A more robust approach would involve saving the columns the imputer was fitted on
                 # and applying imputation only to those columns if they exist in df_processed.

                 # For now, let's apply the imputer's transform method.
                 # This requires the columns in df_processed to be in the same order and count
                 # as the columns the imputer was fitted on. This is a potential mismatch point.

                 # To be more robust, let's fit a *new* imputer only on the columns
                 # that have missing values in the current dataframe. This is a simplification
                 # and assumes the imputation strategy (mean) is appropriate for these columns.
                 # In a real-world scenario, you'd use the *fitted* imputer from training.
                 temp_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                 df_processed[cols_with_missing_processed] = temp_imputer.fit_transform(df_processed[cols_with_missing_processed])


             except Exception as e:
                 st.error(f"An error occurred during imputation: {e}")
                 st.error(traceback.format_exc()) # Print traceback for debugging


        st.write("Processed Data (first 5 rows after cleaning and encoding):")
        st.dataframe(df_processed.head())
        st.write(f"Number of columns after initial processing: {df_processed.shape[1]}")


        # Ensure columns match the training data columns for prediction
        # This is crucial to avoid errors during prediction.
        # We need to ensure the columns in df_processed are exactly the same as
        # the columns the model was trained on, in the same order.

        # Get the list of columns the model was trained on (assuming X from previous steps is available or can be loaded)
        try:
            # This assumes the 'X' variable exists in the kernel's global scope
            # In a deployed app, you would load a saved list of column names
            trained_columns = X.columns.tolist()
            st.write(f"Expected columns based on training data: {len(trained_columns)}")
            # st.write(trained_columns) # Uncomment to see the list of trained columns

        except NameError:
            st.error("Training data columns ('X') not found in the kernel. Cannot align columns for prediction.")
            st.stop()

        # Drop the 'SalePrice' column if it exists in the prediction data
        if 'SalePrice' in df_processed.columns:
            X_predict = df_processed.drop(columns=['SalePrice'])
        else:
            X_predict = df_processed.copy()

        # Align columns of X_predict with the trained columns
        # Add missing columns with a default value (e.g., 0)
        for col in trained_columns:
            if col not in X_predict.columns:
                X_predict[col] = 0 # Add missing columns with 0
                # st.warning(f"Added missing column: {col}")


        # Drop columns in X_predict that were not in the training data
        extra_cols = [col for col in X_predict.columns if col not in trained_columns]
        if extra_cols:
            X_predict = X_predict.drop(columns=extra_cols)
            st.warning(f"Dropped extra columns not in training data: {extra_cols}")


        # Ensure the order of columns is the same as the training data
        X_predict = X_predict[trained_columns]


        st.write("Data ready for prediction (first 5 rows after column alignment):")
        st.dataframe(X_predict.head())
        st.write(f"Number of columns for prediction: {X_predict.shape[1]}")
        # st.write("Columns for prediction:")
        # st.write(X_predict.columns.tolist()) # Uncomment to see the list of prediction columns


        # Select a row for prediction
        if not X_predict.empty:
            row_index = st.number_input("Enter the row index to predict (0-based)", min_value=0, max_value=len(X_predict)-1, value=0)

            if st.button("Predict Price"):
                # Get the selected row
                selected_row = X_predict.iloc[[row_index]]

                # Make prediction
                prediction = model.predict(selected_row)

                st.subheader(f"Predicted House Price for Row {row_index}:")
                st.write(f"${prediction[0]:,.2f}")
        else:
            st.warning("No data available after processing for prediction.")


    except Exception as e:
        st.error(f"An error occurred during processing or prediction: {e}")
        st.error(traceback.format_exc()) # Print traceback for debugging
