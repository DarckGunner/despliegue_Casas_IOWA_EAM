import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the model and encoders
model = joblib.load('gradient_boosting_regressor_tuned.joblib')
onehot_encoder = joblib.load('onehot_encoder.joblib')
label_encoder_kitchenqual = joblib.load('label_encoder_kitchenqual.joblib')

st.title('House Price Prediction')

st.header('Enter House Features:')

# Input fields for features (based on X dataframe columns)
LotFrontage = st.number_input('Lot Frontage', min_value=0.0, value=70.0)
LotArea = st.number_input('Lot Area', min_value=0.0, value=10000.0)
OverallQual = st.slider('Overall Quality', 1, 10, 5)
YearBuilt = st.slider('Year Built', 1800, 2024, 2000)
YearRemodAdd = st.slider('Year Remodeled/Added', 1800, 2024, 2000)
TotalBsmtSF = st.number_input('Total Basement SF', min_value=0.0, value=1000.0)
SecondFlrSF = st.number_input('Second Floor SF', min_value=0.0, value=500.0)
GrLivArea = st.number_input('Above Ground Living Area SF', min_value=0.0, value=1500.0)
FullBath = st.slider('Full Bathrooms', 0, 4, 2)
HalfBath = st.slider('Half Bathrooms', 0, 2, 1)
KitchenQual = st.selectbox('Kitchen Quality', options=list(label_encoder_kitchenqual.keys()))
Fireplaces = st.slider('Number of Fireplaces', 0, 4, 1)
GarageCars = st.slider('Garage Car Capacity', 0, 4, 2)
WoodDeckSF = st.number_input('Wood Deck SF', min_value=0.0, value=100.0)
OpenPorchSF = st.number_input('Open Porch SF', min_value=0.0, value=50.0)

# Input fields for One-Hot Encoded SaleType features
st.subheader('Sale Type:')
SaleType_COD = st.checkbox('Sale Type: C.O.D.')
SaleType_CWD = st.checkbox('Sale Type: CWD')
SaleType_Con = st.checkbox('Sale Type: Con')
SaleType_ConLD = st.checkbox('Sale Type: ConLD')
SaleType_ConLI = st.checkbox('Sale Type: ConLI')
SaleType_ConLw = st.checkbox('Sale Type: ConLw')
SaleType_New = st.checkbox('Sale Type: New')
SaleType_Oth = st.checkbox('Sale Type: Oth')
SaleType_WD = st.checkbox('Sale Type: WD')

# Input fields for One-Hot Encoded SaleCondition features
st.subheader('Sale Condition:')
SaleCondition_Abnorml = st.checkbox('Sale Condition: Abnormal')
SaleCondition_AdjLand = st.checkbox('Sale Condition: Adjacent Land')
SaleCondition_Alloca = st.checkbox('Sale Condition: Allocation')
SaleCondition_Family = st.checkbox('Sale Condition: Family')
SaleCondition_Normal = st.checkbox('Sale Condition: Normal')
SaleCondition_Partial = st.checkbox('Sale Condition: Partial')


if st.button('Predict Price'):
    # Create a dictionary from user inputs
    user_input_dict = {
        'LotFrontage': LotFrontage,
        'LotArea': LotArea,
        'OverallQual': OverallQual,
        'YearBuilt': YearBuilt,
        'YearRemodAdd': YearRemodAdd,
        'TotalBsmtSF': TotalBsmtSF,
        '2ndFlrSF': SecondFlrSF, # Assuming '2ndFlrSF' corresponds to 'Second Floor SF'
        'GrLivArea': GrLivArea,
        'FullBath': FullBath,
        'HalfBath': HalfBath,
        'KitchenQual': label_encoder_kitchenqual[KitchenQual], # Map KitchenQual
        'Fireplaces': Fireplaces,
        'GarageCars': GarageCars,
        'WoodDeckSF': WoodDeckSF,
        'OpenPorchSF': OpenPorchSF,
        # Convert boolean checkboxes to integers for one-hot encoded features
        'SaleType_COD': int(SaleType_COD),
        'SaleType_CWD': int(SaleType_CWD),
        'SaleType_Con': int(SaleType_Con),
        'SaleType_ConLD': int(SaleType_ConLD),
        'SaleType_ConLI': int(SaleType_ConLI),
        'SaleType_ConLw': int(SaleType_ConLw),
        'SaleType_New': int(SaleType_New),
        'SaleType_Oth': int(SaleType_Oth),
        'SaleType_WD': int(SaleType_WD),
        'SaleCondition_Abnorml': int(SaleCondition_Abnorml),
        'SaleCondition_AdjLand': int(SaleCondition_AdjLand),
        'SaleCondition_Alloca': int(SaleCondition_Alloca),
        'SaleCondition_Family': int(SaleCondition_Family),
        'SaleCondition_Normal': int(SaleCondition_Normal),
        'SaleCondition_Partial': int(SaleCondition_Partial)
    }

    # Create a DataFrame from the user input dictionary
    # Ensure column order matches the training data X
    # We need to get the column order from the X dataframe created in the notebook
    # Since X was dropped from df, we need to reconstruct the expected column order
    # based on the notebook's preprocessing steps.
    # The order was: original numerical columns + KitchenQual (encoded) + one-hot encoded columns
    original_numerical_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
                               'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath',
                               'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF'] # Exclude KitchenQual and SalePrice
    encoded_categorical_cols = ['KitchenQual']
    onehot_encoded_cols = onehot_encoder.get_feature_names_out(['SaleType', 'SaleCondition']).tolist()

    # Combine to get the expected order
    expected_column_order = original_numerical_cols + encoded_categorical_cols + onehot_encoded_cols

    # Create the DataFrame with the correct column order
    input_df = pd.DataFrame([user_input_dict])[expected_column_order]

    # Apply imputation (although with current inputs, NaNs are unlikely)
    # Recreate imputer or load if needed (assuming mean imputer from notebook)
    # For robustness, let's define and fit the imputer here if not loaded
    # Based on the notebook, SimpleImputer(missing_values=np.nan, strategy='mean') was used.
    # We need to fit this imputer on the training data columns to ensure it has the correct means.
    # Since we don't have the original training data here, we will skip fitting and assume
    # the imputer would be loaded or fitted on the full training data in a real app.
    # For this task, we will apply imputation directly using a new imputer instance
    # trained on the columns that had NaNs in the original data.
    # This is a simplification for the subtask; in a real app, the fitted imputer should be saved and loaded.

    # Identify columns that had missing values in the original training data (from notebook output)
    cols_with_missing_in_training = ['LotFrontage', 'LotArea', 'YearRemodAdd', 'TotalBsmtSF',
                                     '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath',
                                     'KitchenQual', 'WoodDeckSF', 'OpenPorchSF', 'SaleType', 'SaleCondition']
    # After preprocessing in notebook, KitchenQual, SaleType, SaleCondition are transformed.
    # The columns that needed imputation in the final X were:
    # LotFrontage, LotArea, YearRemodAdd, TotalBsmtSF, 2ndFlrSF, GrLivArea, FullBath, HalfBath,
    # KitchenQual (after mapping), WoodDeckSF, OpenPorchSF, plus the one-hot encoded columns
    # if their original categorical columns had NaNs.

    # Let's re-list the columns in the final X that had NaNs and were imputed in the notebook:
    # LotFrontage, LotArea, YearRemodAdd, TotalBsmtSF, 2ndFlrSF, GrLivArea, FullBath, HalfBath,
    # KitchenQual, WoodDeckSF, OpenPorchSF
    # The one-hot encoded columns derived from SaleType and SaleCondition did not have NaNs after one-hot encoding
    # because the NaNs were in the original categorical columns which were dropped.
    # So, the columns in the final input_df that *could* have NaNs from user input are the original numerical ones,
    # and KitchenQual (though selectbox prevents NaN here).

    # For robustness, let's create an imputer and apply it to the relevant columns.
    # In a real application, you would fit this imputer on the training data and save/load it.
    # Here, we'll apply it to the columns that had NaNs in the notebook's final X.
    cols_for_imputation = ['LotFrontage', 'LotArea', 'YearRemodAdd', 'TotalBsmtSF',
                           '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'KitchenQual',
                           'WoodDeckSF', 'OpenPorchSF']

    # Apply imputation - Note: In a real app, load a pre-fitted imputer and apply it.
    # Since we don't have the fitted imputer object here, we will use a dummy transformation
    # that doesn't change the data but shows the step.
    # A proper implementation requires saving and loading the fitted imputer from the training phase.
    # For the purpose of fulfilling the subtask, we will just show the structure.

    # Create a copy to avoid SettingWithCopyWarning if we were to modify in place
    input_data_processed = input_df.copy()

    # Dummy imputation step - replace with actual imputer.transform in a real app
    # If we were to apply a new imputer fitted on this single row, it would just use the value itself, which is not useful.
    # This step is here to show *where* imputation would happen.

    # For a more realistic representation, let's assume we *can* load a fitted imputer.
    # In a real scenario, the imputer fitted on the training data would be saved.
    # Let's assume we have a loaded_imputer object available.

    # --- Start of Imputation (Conceptual) ---
    # In a real app, you would load the fitted imputer:
    # loaded_imputer = joblib.load('fitted_imputer.joblib')
    # input_data_processed[cols_for_imputation] = loaded_imputer.transform(input_data_processed[cols_for_imputation])
    # --- End of Imputation (Conceptual) ---

    # Since we don't have the fitted imputer file, we skip the actual transformation but acknowledge the step.
    # With the current UI, there are no NaNs to impute anyway.
    # So, input_data_processed is the same as input_df for now.

    # The preprocessed DataFrame is now ready for prediction
    # Storing in input_data_processed as requested.
    # input_data_processed is already created above.

    # Display the preprocessed data (optional, for debugging)
    # st.write("Preprocessed Input Data:")
    # st.write(input_data_processed)

    # The next step will be to make the prediction using `model.predict(input_data_processed)`
    pass # Placeholder for prediction logic
