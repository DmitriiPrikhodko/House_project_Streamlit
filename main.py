import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn

from sklearn.metrics import *
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OrdinalEncoder,
    TargetEncoder,
    FunctionTransformer,
)
import warnings

warnings.filterwarnings("ignore")

sklearn.set_config(transform_output="pandas")

from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

# for model learning
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)


def float_to_int(x):
    return x.astype(int)


def new_features(X):
    X["TotalBsmtBath"] = (
        X["BsmtFullBath"] + X["BsmtHalfBath"] * 0.5
    )  # считаем сколько санузлов в подвале, HalfBath умножили на 0.5, потому что это половина)))
    X["TotalBath"] = X["FullBath"] + X["HalfBath"] * 0.5  # то же самое
    X["TotalSquare"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
    X["GarageAgeYrs"] = X["YrSold"] - X["GarageYrBlt"]
    X["TotalLivArea"] = (
        X["GrLivArea"] + X["BsmtFinSF1"] + X["BsmtFinSF2"]
    )  # считаем общую жилую площадь
    X["TotalRooms"] = X["TotRmsAbvGrd"] + X["BedroomAbvGr"]
    X["Age"] = X["YrSold"] - X["YearBuilt"]
    X["Garage"] = X["GarageCars"].map(lambda x: x if x == 0 else 1)
    X["YrsFromRenn"] = X["YrSold"] - X["YearRemodAdd"]
    return X


def processing(X):
    X = new_features(X)
    num_features = [
        "LotFrontage",
        "LotArea",
        "YearBuilt",
        "YearRemodAdd",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "GarageYrBlt",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "YrSold",
        "TotalSquare",
        "TotalBath",
        "TotalBsmtBath",
        "GarageAgeYrs",
        "Age",
        "TotalRooms",
        "TotalLivArea",
        "YrsFromRenn",
    ]
    cat_features = []
    for c in X.columns:
        if c not in num_features and c != "Id":
            cat_features.append(c)
    drop_features = [
        "PoolArea",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MasVnrType",
        "Alley",
        "OverallCond",
        "GarageQual",
        "GarageCond",
        "LandContour",
        "MSSubClass",
        "SaleType",
        "LowQualFinSF",
        "LandSlope",
        "MiscVal",
        "Utilities",
        "GarageArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GarageType",
        "GarageYrBlt",
        "GarageFinish",
        "GarageArea",
        "GarageQual",
        "GarageCond",
        "GrLivArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "TotRmsAbvGrd",
        "BedroomAbvGr",
        "YrSold",
        "YearBuilt",
        "YearRemodAdd",
    ]
    rest_cat_features = [
        c for c in cat_features if c not in drop_features
    ]  # все недропнутые категориальные
    rest_cat_num_features = [
        c for c in rest_cat_features if X[c].dtypes == int
    ]  # отбираем те, которые с цифрами
    rest_cat_str_features = [
        c for c in rest_cat_features if c not in rest_cat_num_features
    ]  # отбираем те,которые с буквами
    rest_num_features = [
        c for c in num_features if c not in drop_features
    ]  # числовые признаки
    imputer = ColumnTransformer(
        transformers=[
            ("drop_features", "drop", drop_features),
            (
                "imput_no",
                SimpleImputer(strategy="constant", fill_value="no"),
                rest_cat_str_features,
            ),  # type: ignore
            (
                "imput_0",
                SimpleImputer(strategy="constant", fill_value=0),
                rest_cat_num_features,
            ),
            (
                "imput_numeric",
                SimpleImputer(strategy="median"),
                rest_num_features,
                # [c for c in num_features if c not in drop_features],
            ),
        ],
        verbose_feature_names_out=False,
        remainder="passthrough",
    )
    scaler = ColumnTransformer(
        [("scaling_num_columns", StandardScaler(), rest_num_features)],
        verbose_feature_names_out=False,
        remainder="passthrough",
    )

    to_int = FunctionTransformer(float_to_int)

    encoder = ColumnTransformer(
        [("encoding", make_pipeline(OrdinalEncoder(), to_int), rest_cat_features)],
        verbose_feature_names_out=False,
        remainder="passthrough",
    )
    preprocessor = Pipeline(
        [
            ("imputer", imputer),
            ("encoder", encoder),
            ("scaler", scaler),
        ]
    )
    X_preproc = preprocessor.fit_transform(X)
    return X_preproc


def reset_form():
    st.session_state.prediction_made = False
    st.session_state.prediction_result = None
    st.session_state.show_form = True
    st.session_state.file_uploaded = False


@st.cache_data
def upload_file(file):
    df = pd.read_csv(file, index_col=False)
    return df


st.title(":blue[Предсказание стоимости дома]")
st.write(
    """
#### _Приходько Дмитрий, Алекберов Марат, Надежда Ишпайкина, Ахмед Хулаев_
"""
)


if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "show_form" not in st.session_state:
    st.session_state.show_form = True
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "test_df" not in st.session_state:
    st.session_state.test_df = None

if not st.session_state.prediction_made and st.session_state.show_form:
    get_file = st.file_uploader("Загрузи CSV файл с данными о недвижимости")

    if get_file is not None:
        st.session_state.test_df = upload_file(get_file)
        st.session_state.file_uploaded = True
    else:
        st.stop()

if st.session_state.file_uploaded == True:
    length = st.session_state.test_df.shape[0]
    st.write("#### Загруженные данные")
    if length > 5:
        st.dataframe(st.session_state.test_df)
    else:
        st.dataframe(st.session_state.test_df.T)

    if st.button("Рассчитать цену"):
        try:
            test_preproc = processing(new_features(st.session_state.test_df.copy()))
            model = joblib.load("model/model.pkl")
            prediction = np.round(np.exp(model.predict(test_preproc)))
            if length == 1:
                st.session_state.prediction_result = prediction[0]
            else:
                st.session_state.prediction_result = prediction
            st.session_state.prediction_made = True
            st.session_state.show_form = False

        except Exception as e:
            st.error(f"Error loading model or making prediction: {e}")

if st.session_state.prediction_made:
    if length == 1:
        st.success(
            f"Предполагаем цену: {np.round(st.session_state.prediction_result)} USD"
        )
    else:
        st.success("Предполагаемы цены:")
        st.dataframe(
            pd.DataFrame(
                {
                    "Id": st.session_state.test_df["Id"],
                    "Price Prediction (USD)": st.session_state.prediction_result,
                }
            )
        )

    # Кнопка для нового предсказания
    if st.button("Новый расчет"):
        predictions = None
        reset_form()
        st.rerun()  # Перезагружаем страницу для очистки формы
