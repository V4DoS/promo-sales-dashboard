import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title="Прогноз Промо-Продаж (LightGBM)", layout="wide", initial_sidebar_state="expanded")

# Кастом CSS для красоты
st.markdown("""
<style>
.metric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
.stButton > button {background-color: #4ECDC4; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Дашборд: Прогноз промо-продаж (LightGBM)")

# Сайдбар для настроек
st.sidebar.header("⚙️ Настройки модели")
N_FOLDS = st.sidebar.slider("Количество фолдов (KFold):", min_value=3, max_value=10, value=5)
FORECAST_DAYS = st.sidebar.slider("Дни прогноза:", min_value=7, max_value=90, value=30)
lags_input = st.sidebar.text_input("LAGS (через запятую, напр. 7,14,28):", value="7,14,28")
ROLL_WINDOWS_input = st.sidebar.text_input("ROLL_WINDOWS (через запятую, напр. 7,14,28):", value="7,14,28")
LAGS = [int(x.strip()) for x in lags_input.split(',') if x.strip().isdigit()]
ROLL_WINDOWS = [int(x.strip()) for x in ROLL_WINDOWS_input.split(',') if x.strip().isdigit()]
RANDOM_STATE = st.sidebar.number_input("Random State:", value=42)

if st.sidebar.button("🔄 Обновить данные и модель"):
    st.rerun()

# Кэшированная загрузка и подготовка данных
@st.cache_data
def load_and_prepare_data():
    try:
        INPUT_FILE = 'Продажи1.xlsx'
        SALES_SHEET = 'Продажи'
        PRICES_SHEET = 'Цены'
        
        sales = pd.read_excel(INPUT_FILE, sheet_name=SALES_SHEET)
        promo_prices = pd.read_excel(INPUT_FILE, sheet_name=PRICES_SHEET)
        
        # Очистка названий столбцов
        sales.columns = sales.columns.astype(str).str.strip()
        promo_prices.columns = promo_prices.columns.astype(str).str.strip()
        
        # Автоматическое переименование по ключевым словам
        rename_map_sales = {}
        for c in sales.columns:
            cl = c.lower()
            if 'дат' in cl:
                rename_map_sales[c] = 'ds'
            elif 'номенк' in cl:
                rename_map_sales[c] = 'SKU'
            elif 'кол' in cl:
                rename_map_sales[c] = 'qty'
            elif 'цен' in cl:
                rename_map_sales[c] = 'price'
        sales = sales.rename(columns=rename_map_sales)
        
        rename_map_promo = {}
        for c in promo_prices.columns:
            cl = c.lower()
            if 'номенк' in cl:
                rename_map_promo[c] = 'SKU'
            elif 'цен' in cl:
                rename_map_promo[c] = 'promo_price'
        promo_prices = promo_prices.rename(columns=rename_map_promo)
        
        # Проверка обязательных колонок
        required_sales_cols = {'ds', 'SKU', 'qty', 'price'}
        if not required_sales_cols.issubset(sales.columns):
            raise ValueError(f"Не найдены нужные колонки в листе 'Продажи': {sales.columns.tolist()}")
        required_price_cols = {'SKU', 'promo_price'}
        if not required_price_cols.issubset(promo_prices.columns):
            raise ValueError(f"Не найдены нужные колонки в листе 'Цены': {promo_prices.columns.tolist()}")
        
        # Приведение типов
        sales['ds'] = pd.to_datetime(sales['ds'])
        sales['SKU'] = sales['SKU'].astype(str)
        promo_prices['SKU'] = promo_prices['SKU'].astype(str)
        
        # Агрегация
        sales = sales.groupby(['ds', 'SKU'], as_index=False).agg({'qty':'sum','price':'mean'})
        
        # Заполнение пропусков по датам
        skus = sales['SKU'].unique()
        min_date = sales['ds'].min()
        max_date = sales['ds'].max()
        all_dates = pd.date_range(min_date, max_date)
        df_full = pd.MultiIndex.from_product([skus, all_dates], names=['SKU', 'ds']).to_frame(index=False)
        df = df_full.merge(sales, on=['SKU','ds'], how='left')
        df = df.sort_values(['SKU','ds'])
        df['qty'] = df['qty'].fillna(0)
        df['price'] = df.groupby('SKU')['price'].ffill().bfill()
        df['price'] = df['price'].fillna(df['price'].mean())
        
        # Временные признаки
        df['dow'] = df['ds'].dt.weekday
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
        
        # Лаги и скользящие средние
        for lag in LAGS:
            df[f'lag_{lag}'] = df.groupby('SKU')['qty'].shift(lag).fillna(0)
        for w in ROLL_WINDOWS:
            df[f'roll_mean_{w}'] = (
                df.groupby('SKU')['qty']
                .shift(1)
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
        
        # Относительная цена
        mean_price_per_sku = df.groupby('SKU')['price'].transform('mean')
        df['price_rel'] = df['price'] / (mean_price_per_sku + 1e-9)
        
        # Кодирование SKU
        le = LabelEncoder()
        df['SKU_le'] = le.fit_transform(df['SKU'])
        
        FEATURES = ['price','price_rel','dow','month','day','is_weekend','SKU_le'] + \
                   [f'lag_{l}' for l in LAGS] + [f'roll_mean_{w}' for w in ROLL_WINDOWS]
        TARGET = 'qty'
        
        return df, promo_prices, le, FEATURES, TARGET, min_date, max_date
    
    except Exception as e:
        st.error(f"Ошибка подготовки данных: {e}")
        return None, None, None, None, None, None, None

# Автоматическая загрузка данных
df, promo_prices, le, FEATURES, TARGET, min_date, max_date = load_and_prepare_data()

if df is not None:
    # Обучение модели (с прогрессом)
    st.subheader("🧠 Обучение модели")
    progress_bar = st.progress(0)
    
    X = df[FEATURES]
    y = df[TARGET]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = []
    oof_preds = np.zeros(len(df))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_data = lgb.Dataset(X_tr