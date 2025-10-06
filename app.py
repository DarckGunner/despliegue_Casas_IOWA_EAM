import os
# Evita el error "inotify watch limit reached" en Streamlit Cloud
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# =============================
# Cargar el modelo y los encoders
# =============================
model = joblib.load('gradient_boosting_regressor_tuned.joblib')
onehot_encoder = joblib.load('onehot_encoder.joblib')
label_encoder_kitchenqual = joblib.load('label_encoder_kitchenqual.joblib')

st.title('🏠 House Price Prediction App')
st.write("Ingresa las características de la vivienda para estimar su precio:")

# =============================
# Entradas del usuario
# =============================
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

# =============================
# Sale Type (One-Hot Encoded)
# =============================
st.subheader('Sale Type:')
SaleType_COD = st.checkbox('C.O.D.')
SaleType_CWD = st.checkbox('CWD')
SaleType_Con = st.checkbox('Con')
SaleType_ConLD = st.checkbox('ConLD')
SaleType_ConLI = st.checkbox('ConLI')
SaleType_ConLw = st.checkbox('ConLw')
SaleType_New = st.checkbox('New')
SaleType_Oth = st.checkbox('Oth')
SaleType_WD = st.checkbox('WD')

# =============================
# Sale Condition (One-Hot Encoded)
# =============================
st.subheader('Sale Condition:')
SaleCondition_Abnorml = st.checkbox('Abnormal')
SaleCondition_AdjLand = st.checkbox('Adjacent Land')
SaleCondition_Alloca = st.checkbox('Allocation')
SaleCondition_Family = st.checkbox('Family')
SaleCondition_Normal = st.checkbox('Normal')
SaleCondition_Partial = st.checkbox('Partial')

# =============================
# Botón de predicción
# =============================
if st.button('🔮 Predict Price'):
    # Crear diccionario con los datos del usuario
    user_input_dict = {
        'LotFrontage': LotFrontage,
        'LotArea': LotArea,
        'OverallQual': OverallQual,
        'YearBuilt': YearBuilt,
        'YearRemodAdd': YearRemodAdd,
        'TotalBsmtSF': TotalBsmtSF,
        '2ndFlrSF': SecondFlrSF,
        'GrLivArea': GrLivArea,
        'FullBath': FullBath,
        'HalfBath': HalfBath,
        'KitchenQual': label_encoder_kitchenqual[KitchenQual],
        'Fireplaces': Fireplaces,
        'GarageCars': GarageCars,
        'WoodDeckSF': WoodDeckSF,
        'OpenPorchSF': OpenPorchSF,
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

    # Columnas esperadas (orden correcto)
    original_numerical_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd',
                               'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath',
                               'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF']
    encoded_categorical_cols = ['KitchenQual']
    onehot_encoded_cols = onehot_encoder.get_feature_names_out(['SaleType', 'SaleCondition']).tolist()
    expected_column_order = original_numerical_cols + encoded_categorical_cols + onehot_encoded_cols

    # Crear DataFrame con el orden correcto
    input_df = pd.DataFrame([user_input_dict])[expected_column_order]

    # Copiar a variable procesada (en este caso no hay imputación real)
    input_data_processed = input_df.copy()

    # =============================
    # Predicción
    # =============================
    try:
        prediction = model.predict(input_data_processed)[0]
        st.success(f"💰 Estimated House Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

    # Mostrar los datos ingresados (opcional, útil para depuración)
    with st.expander("Ver datos procesados utilizados en la predicción"):
        st.dataframe(input_data_processed)
