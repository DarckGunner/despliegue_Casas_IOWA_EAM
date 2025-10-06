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

# Introduction and Instructions
st.markdown("""
This application predicts house prices based on various features.
Please enter the details of the house in the input fields below to get a price prediction.
""")

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
    original_numerical_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
                               'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath',
                               'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF']
    encoded_categorical_cols = ['KitchenQual']
    onehot_encoded_cols = onehot_encoder.get_feature_names_out(['SaleType', 'SaleCondition']).tolist()

    expected_column_order = original_numerical_cols + encoded_categorical_cols + onehot_encoded_cols

    input_df = pd.DataFrame([user_input_dict])[expected_column_order]

    # Apply imputation (conceptual - see previous notes)
    # In a real app, load a pre-fitted imputer.
    # For this task, we acknowledge the step but don't perform a functional imputation
    # as current UI prevents NaNs.
    input_data_processed = input_df.copy() # No change with current UI

    # Make prediction
    prediction = model.predict(input_data_processed)

    # Display the prediction
    st.subheader('Predicted House Price:')
    st.write(f'${prediction[0]:,.2f}')

# Information about the model
st.sidebar.header("About the Model")
st.sidebar.info("""
This application uses a Gradient Boosting Regressor model trained on the Iowa House Prices dataset.
The model was trained to predict the `SalePrice` based on various house features.
""")

# Limitations or Assumptions
st.sidebar.header("Assumptions and Limitations")
st.sidebar.warning("""
*   The predictions are based on the patterns learned from the training data and may not be accurate for houses with features significantly different from those in the dataset.
*   The app assumes that the user provides realistic inputs for the house features.
*   The model's performance is limited by the quality and representativeness of the training data.
""")
