import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

KGM_TO_NM = 9.80665

class CleaningTransformer(BaseEstimator, TransformerMixin):
    numeric_cols = ["mileage", "engine", "max_power", "torque", "max_torque_rpm", "seats"]

    def clean_numeric(self, value):
        if pd.isna(value):
            return np.nan
        value = str(value)
        num = re.findall(r"\d+\.?\d*", value)
        return float(num[0]) if num else np.nan

    def parse_torque(self, text):
        if pd.isna(text):
            return np.nan, np.nan
        text = str(text).lower()
        nm_match = re.search(r"(\d+\.?\d*)\s*nm", text)
        kgm_match = re.search(r"(\d+\.?\d*)\s*kgm", text)
        at_match = re.search(r"^(\d+\.?\d*)\s*@", text)
        torque_nm = None
        if nm_match:
            torque_nm = float(nm_match.group(1))
        elif kgm_match:
            torque_nm = float(kgm_match.group(1)) * KGM_TO_NM
        elif at_match:
            torque_nm = float(at_match.group(1)) * KGM_TO_NM

        rpm_match = re.search(r"(\d[\d,]*)(?:\s*-\s*(\d[\d,]*))?\s*rpm", text)
        if not rpm_match:
            rpm_match = re.search(r"@?\s*(\d[\d,]*)", text)
        rpm = None
        if rpm_match:
            rpm1 = rpm_match.group(1).replace(",", "")
            rpm2 = rpm_match.group(2).replace(",", "") if rpm_match.lastindex == 2 else None
            try:
                rpm = int(rpm1)
                if rpm2:
                    rpm = max(rpm, int(rpm2))
            except:
                pass
        return torque_nm, rpm

    def fit(self, X, y=None):
        self.medians_ = X[self.numeric_cols].apply(pd.to_numeric, errors="coerce").median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in ["mileage", "engine", "max_power"]:
            X[col] = X[col].apply(self.clean_numeric)
        X["torque"], X["max_torque_rpm"] = zip(*X["torque"].apply(self.parse_torque))
        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.medians_[col])
        X["engine"] = X["engine"].astype(int)
        X["seats"] = X["seats"].astype(int)
        return X

class NameParser(BaseEstimator, TransformerMixin):
    def parse_name(self, name):
        tokens = str(name).split()
        brand = tokens[0]
        model = tokens[1] if len(tokens) > 1 else None
        fuel = None
        for f in ["Diesel", "Petrol", "CNG", "LPG", "Electric"]:
            if f.lower() in str(name).lower():
                fuel = f
                break
        engine_capacity = None
        match = re.search(r"\b\d\.\d\b", str(name))
        if match:
            engine_capacity = match.group()
        tail_tokens = tokens[2:]
        tail_tokens = [t for t in tail_tokens if t not in (fuel, engine_capacity)]
        config = " ".join(tail_tokens) if tail_tokens else None
        return pd.Series([brand, model, fuel, engine_capacity, config])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        name_features = X["name"].apply(self.parse_name)
        name_features.columns = ["brand", "model", "fuel", "engine_capacity", "config"]
        X = pd.concat([X.drop(columns=[c for c in name_features.columns if c in X.columns]), name_features], axis=1)
        return X

@st.cache_resource
def load_pipeline():
    return joblib.load("best_model.pkl")

pipeline = load_pipeline()

st.title("Предсказание стоимости автомобиля (Ridge + OHE)")

st.header("📂 Быстрый EDA по готовым данным")

col1, col2 = st.columns(2)

if col1.button("🔍 Показать EDA по TRAIN"):
    try:
        df_train_local = pd.read_csv("data/df_train.csv")
        del df_train_local['Unnamed: 0']
        st.success("Загружен df_train.csv")
        
        st.subheader("Первые строки train")
        st.write(df_train_local.head())

        num_cols_tr = df_train_local.select_dtypes(include=np.number).columns.tolist()
        cat_cols_tr = df_train_local.select_dtypes(include="object").columns.tolist()

        st.subheader("Распределение числового признака (train)")
        if num_cols_tr:
            selected = st.selectbox("Выберите числовой признак", num_cols_tr, key="train_num")
            st.bar_chart(df_train_local[selected])

        st.subheader("Распределение категориального признака (train)")
        if cat_cols_tr:
            selected = st.selectbox("Выберите категориальный признак", cat_cols_tr, key="train_cat")
            st.bar_chart(df_train_local[selected].value_counts())

        if len(num_cols_tr) > 1:
            st.subheader("Корреляционная матрица (train)")
            corr = df_train_local[num_cols_tr].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка загрузки train: {e}")

if col2.button("🔍 Показать EDA по TEST"):
    try:
        df_test_local = pd.read_csv("data/df_test.csv")
        del df_test_local['Unnamed: 0']
        
        st.success("Загружен df_test.csv")

        st.subheader("Первые строки test")
        st.write(df_test_local.head())

        num_cols_ts = df_test_local.select_dtypes(include=np.number).columns.tolist()
        cat_cols_ts = df_test_local.select_dtypes(include="object").columns.tolist()

        st.subheader("Распределение числового признака (test)")
        if num_cols_ts:
            selected = st.selectbox("Выберите числовой признак", num_cols_ts, key="test_num")
            st.bar_chart(df_test_local[selected])

        st.subheader("Распределение категориального признака (test)")
        if cat_cols_ts:
            selected = st.selectbox("Выберите категориальный признак", cat_cols_ts, key="test_cat")
            st.bar_chart(df_test_local[selected].value_counts())

        if len(num_cols_ts) > 1:
            st.subheader("Корреляционная матрица (test)")
            corr = df_test_local[num_cols_ts].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка загрузки test: {e}")

uploaded_data = st.file_uploader("Загрузите CSV-файл для анализа и предсказаний", type=["csv"], key="eda_and_predict")
df = None
if uploaded_data is not None:
    try:
        df = pd.read_csv(uploaded_data)
        st.success(f"Файл загружен: {uploaded_data.name}")
    except Exception as e:
        st.error(f"Ошибка при чтении CSV: {e}")

if df is not None:
    name_parser = NameParser()
    df = name_parser.fit_transform(df)

    st.subheader("Первые строки")
    st.write(df.head())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.subheader("Гистограммы числовых признаков")
    selected_num = st.selectbox("Выберите числовой признак", num_cols, key="num_col")
    if selected_num:
        st.bar_chart(df[selected_num].dropna())

    st.subheader("Категориальные признаки")
    selected_cat = st.selectbox("Выберите категориальный признак", cat_cols, key="cat_col")
    st.bar_chart(df[selected_cat].value_counts())
    
    if cat_cols:
        st.subheader("📉 Boxplot числового признака по категориальному")
        selected_num_box = st.selectbox("Выберите числовой признак для boxplot", num_cols, key="box_num")
        selected_cat_box = st.selectbox("Выберите категориальный признак для boxplot", cat_cols, key="box_cat")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=selected_cat_box, y=selected_num_box, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if len(num_cols) > 1:
        st.subheader("📈 Корреляционная матрица числовых признаков")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    if "price" in df.columns:
        st.subheader("💰 Распределение цены")
        fig, ax = plt.subplots()
        sns.histplot(df["price"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("❗ Пропущенные значения")
    missing = df.isna().sum()
    st.bar_chart(missing[missing > 0])

    if st.button("Предсказать цены для всех строк файла", key="predict_all"):
        try:
            preds = pipeline.predict(df)
            df["predicted_price"] = preds
            st.subheader("Результаты предсказания")
            st.write(df)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

st.header("🎛 Ручной ввод признаков")
manual_data = {}
manual_data["name"] = st.text_input("Введите полное название автомобиля (name)", value="", key="manual_name")

numeric_cols = ["mileage", "engine", "max_power", "torque", "max_torque_rpm", "seats"]
for col in numeric_cols:
    manual_data[col] = st.number_input(col, value=0.0, key=f"manual_{col}")

cat_cols = ["seller_type", "transmission", "owner"]
for col in cat_cols:
    options = df[col].dropna().unique().tolist() if df is not None else []
    selected = st.selectbox(f"{col} (выберите из списка)", options, key=f"manual_{col}_select")
    manual_input = st.text_input(f"Или введите своё значение для {col}", value="", key=f"manual_{col}_input")
    manual_data[col] = manual_input.strip() if manual_input.strip() != "" else selected

if st.button("Предсказать", key="manual_predict"):
    if manual_data["name"].strip() == "":
        st.warning("Введите название автомобиля для предсказания.")
    else:
        df_manual = pd.DataFrame([manual_data])
        pred = pipeline.predict(df_manual)[0]
        st.success(f"💰 Предсказанная цена: **{pred:,.0f}** INR")

st.header("📈 Веса Ridge-модели")

coefs = pipeline.named_steps["model"].coef_
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
coef_df = pd.DataFrame({"feature": feature_names, "weight": coefs})
coef_df = coef_df.sort_values("weight", key=np.abs, ascending=False).head(30)

fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(data=coef_df, x="weight", y="feature", ax=ax)
st.pyplot(fig)