# -*- coding: utf-8 -*-
"""
Прогноз промо-продаж (LightGBM, глобальная модель)
Файл: Продажи1.xlsx
Листы: "Продажи" и "Цены"
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from datetime import timedelta
from sklearn.metrics import mean_squared_error

# ------------------ Параметры ------------------
INPUT_FILE = r'Продажи1.xlsx'   # путь к вашему файлу
SALES_SHEET = 'Продажи'
PRICES_SHEET = 'Цены'
N_FOLDS = 5
FORECAST_DAYS = 30
LAGS = [7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]
RANDOM_STATE = 42
# ------------------------------------------------

# === 1. Загрузка ===
sales = pd.read_excel(INPUT_FILE, sheet_name=SALES_SHEET)
promo_prices = pd.read_excel(INPUT_FILE, sheet_name=PRICES_SHEET)

# === 2. Очистка названий столбцов ===
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

# === 3. Приведение типов ===
sales['ds'] = pd.to_datetime(sales['ds'])
sales['SKU'] = sales['SKU'].astype(str)
promo_prices['SKU'] = promo_prices['SKU'].astype(str)

# === 4. Агрегация ===
sales = sales.groupby(['ds', 'SKU'], as_index=False).agg({'qty':'sum','price':'mean'})

# === 5. Заполнение пропусков по датам ===
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

# === 6. Временные признаки ===
df['dow'] = df['ds'].dt.weekday
df['month'] = df['ds'].dt.month
df['day'] = df['ds'].dt.day
df['is_weekend'] = df['dow'].isin([5,6]).astype(int)

# === 7. Лаги и скользящие средние ===
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

# === 8. Обучение модели ===
X = df[FEATURES]
y = df[TARGET]

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
models = []
oof_preds = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective':'regression',
        'metric':'rmse',
        'verbosity': -1,
        'seed': RANDOM_STATE + fold,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }

    # --- Исправленная часть ---
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)  # каждые 100 итераций вывод
        ]
    )

    models.append(model)
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

rmse = mean_squared_error(y, oof_preds, squared=False)
print(f'OOF RMSE: {rmse:.4f}')

# === 9. Прогноз на новые цены ===
forecast_start = max_date + timedelta(days=1)
forecast_dates = pd.date_range(forecast_start, forecast_start + timedelta(days=FORECAST_DAYS-1))

rows = []
for _, promo_row in promo_prices.iterrows():
    sku = promo_row['SKU']
    promo_price = promo_row['promo_price']
    last = df[df['SKU'] == sku].sort_values('ds').iloc[-1] if sku in df['SKU'].values else None

    for ds in forecast_dates:
        if last is not None:
            base = last.copy()
            entry = {
                'ds': ds,
                'SKU': sku,
                'price': promo_price,
                'price_rel': promo_price / (base['price'] + 1e-9),
                'dow': ds.weekday(),
                'month': ds.month,
                'day': ds.day,
                'is_weekend': int(ds.weekday() in [5,6]),
                'SKU_le': le.transform([sku])[0] if sku in le.classes_ else int(np.median(le.transform(le.classes_)))
            }
            for l in LAGS:
                entry[f'lag_{l}'] = base.get(f'lag_{l}', 0)
            for w in ROLL_WINDOWS:
                entry[f'roll_mean_{w}'] = base.get(f'roll_mean_{w}', 0)
        else:
            entry = {
                'ds': ds, 'SKU': sku, 'price': promo_price, 'price_rel': 1,
                'dow': ds.weekday(), 'month': ds.month, 'day': ds.day,
                'is_weekend': int(ds.weekday() in [5,6]),
                'SKU_le': int(np.median(le.transform(le.classes_)))
            }
            for l in LAGS:
                entry[f'lag_{l}'] = 0
            for w in ROLL_WINDOWS:
                entry[f'roll_mean_{w}'] = 0

        rows.append(entry)

pred_df = pd.DataFrame(rows)
X_pred = pred_df[FEATURES]

preds = np.zeros(len(X_pred))
for model in models:
    preds += model.predict(X_pred, num_iteration=model.best_iteration) / len(models)

pred_df['pred_qty'] = np.clip(preds, 0, None)

# === 10. Сохранение результата ===
pred_df[['ds','SKU','pred_qty']].to_excel('promo_forecast.xlsx', index=False)
print("✅ Готово! Прогноз сохранён в 'promo_forecast.xlsx'")
