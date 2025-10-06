import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Cargar los objetos entrenados
try:
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    label_encoder_kitchenqual = joblib.load('label_encoder_kitchenqual.joblib')
    model = joblib.load('gradient_boosting_regressor_tuned.joblib')
    imputer = joblib.load('knn_imputer_final.joblib')

except FileNotFoundError:
    st.error("¡Error al cargar los archivos necesarios! Asegúrate de que 'onehot_encoder.joblib', 'label_encoder_kitchenqual.joblib', 'gradient_boosting_regressor_tuned.joblib' y 'knn_imputer_final.joblib' estén en el mismo directorio.")
    st.stop()

# Definir las columnas a eliminar (las mismas que en el notebook)
columns_to_drop = ['Id', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold']


st.title("Aplicación de Predicción de Valor de Casas")

uploaded_file = st.file_uploader("Carga tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    # Mostrar las primeras filas del CSV cargado
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df_input.head())

    # Permitir al usuario seleccionar una fila por ID
    if 'Id' in df_input.columns:
        house_ids = df_input['Id'].tolist()
        selected_id = st.selectbox("Selecciona el ID de la casa para predecir:", house_ids)

        if st.button("Realizar Predicción"):
            # Seleccionar la fila correspondiente al ID
            row_to_predict = df_input[df_input['Id'] == selected_id].copy()

            # Aplicar el mismo preprocesamiento que en el notebook
            # Eliminar columnas
            row_to_predict = row_to_predict.drop(columns=[col for col in columns_to_drop if col in row_to_predict.columns])

            # Codificar KitchenQual con manejo de categorías no vistas
            known_classes = set(label_encoder_kitchenqual.classes_)
            mask_valid = row_to_predict['KitchenQual'].isin(known_classes)

            if not mask_valid.all():
                default_class = label_encoder_kitchenqual.classes_[-1]
                row_to_predict.loc[~mask_valid, 'KitchenQual'] = default_class

            row_to_predict['KitchenQual'] = label_encoder_kitchenqual.transform(row_to_predict['KitchenQual'])

            # Codificación One-Hot para SaleType y SaleCondition
            columns_to_encode = ['SaleType', 'SaleCondition']
            try:
                encoded_data = onehot_encoder.transform(row_to_predict[columns_to_encode])
                encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(columns_to_encode))

                # Eliminar las columnas originales y concatenar las codificadas
                row_to_predict = row_to_predict.drop(columns=[col for col in columns_to_encode if col in row_to_predict.columns])
                row_to_predict = pd.concat([row_to_predict, encoded_df], axis=1)

            except ValueError as e:
                 st.error(f"Error durante la codificación One-Hot: {e}. Asegúrate de que las columnas 'SaleType' y 'SaleCondition' en tu CSV contengan solo categorías vistas durante el entrenamiento del codificador.")
                 st.stop()


            # Eliminar la columna 'SalePrice' si existe antes de imputar y predecir
            if 'SalePrice' in row_to_predict.columns:
                X_predict = row_to_predict.drop(columns=['SalePrice'])
            else:
                X_predict = row_to_predict

            # Asegurarse de que las columnas de X_predict coincidan con las columnas usadas para entrenar el imputador y el modelo
            # Esto es crucial si el CSV de entrada no tiene todas las columnas originales (excepto las dropeadas y SalePrice)
            # Obtener las columnas esperadas por el imputador y el modelo
            expected_columns = imputer.feature_names_in_ # O model.feature_names_in_ si tu modelo tiene este atributo

            # Rellenar las columnas faltantes con 0s (o NaN si el imputador lo maneja) y reordenar
            for col in expected_columns:
                if col not in X_predict.columns:
                    X_predict[col] = 0 # O np.nan si el imputador maneja NaN

            X_predict = X_predict[expected_columns]


            # Aplicar el imputador
            X_predict_imputed = pd.DataFrame(imputer.transform(X_predict), columns=X_predict.columns)


            # Realizar la predicción
            prediction = model.predict(X_predict_imputed)

            st.subheader(f"Predicción para la casa con ID {selected_id}:")
            st.write(f"El valor de predicción es: ${prediction[0]:,.2f}")

    else:
        st.warning("El archivo CSV cargado no contiene una columna 'Id'. No se puede seleccionar una fila por ID.")
