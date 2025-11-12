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
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)  # Без логов для дашборда
            ]
        )
        models.append(model)
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        progress_bar.progress((fold / N_FOLDS) * 100)
    
    rmse = mean_squared_error(y, oof_preds, squared=False)
    st.metric("OOF RMSE (точность модели)", f"{rmse:.4f}")
    progress_bar.progress(100)
    
    # Выбор SKU для просмотра
    unique_skus = list(df['SKU'].unique())
    selected_sku = st.selectbox("🛒 Выберите SKU для просмотра:", unique_skus)
    
    # Получение наименования (если есть)
    if 'Наименование' in promo_prices.columns:
        name_row = promo_prices[promo_prices['SKU'] == selected_sku]
        if not name_row.empty:
            product_name = name_row['Наименование'].iloc[0]
            st.markdown(f"**📝 Наименование SKU:** {product_name}")
    
    # Слайдер для promo_price
    promo_row = promo_prices[promo_prices['SKU'] == selected_sku]
    if not promo_row.empty:
        default_promo_price = promo_row['promo_price'].iloc[0]
    else:
        default_promo_price = df[df['SKU'] == selected_sku]['price'].mean()
    promo_price = st.slider("💰 Promo-цена для SKU (₽):", min_value=0.0, max_value=default_promo_price * 2, value=default_promo_price, step=0.1)
    
    # Кнопки сценариев
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    avg_price_sku = df[df['SKU'] == selected_sku]['price'].mean()
    if col_btn1.button("🔥 Скидка 10%"):
        promo_price = avg_price_sku * 0.9
        st.rerun()
    if col_btn2.button("💥 Скидка 20%"):
        promo_price = avg_price_sku * 0.8
        st.rerun()
    if col_btn3.button("➡️ Средняя цена"):
        promo_price = avg_price_sku
        st.rerun()
    
    # Прогноз на новые цены
    forecast_start = max_date + timedelta(days=1)
    forecast_dates = pd.date_range(forecast_start, forecast_start + timedelta(days=FORECAST_DAYS-1))
    rows = []
    for _, promo_row in promo_prices.iterrows():
        sku = promo_row['SKU']
        this_promo_price = promo_price if sku == selected_sku else promo_row['promo_price']  # Только для выбранного SKU меняем цену
        last = df[df['SKU'] == sku].sort_values('ds').iloc[-1] if sku in df['SKU'].values else None
        for ds in forecast_dates:
            if last is not None:
                base = last.copy()
                entry = {
                    'ds': ds,
                    'SKU': sku,
                    'price': this_promo_price,
                    'price_rel': this_promo_price / (base['price'] + 1e-9),
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
                    'ds': ds, 'SKU': sku, 'price': this_promo_price, 'price_rel': 1,
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
    
    # Фильтр для выбранного SKU
    sku_forecast = pred_df[pred_df['SKU'] == selected_sku][['ds', 'pred_qty']].copy()
    sku_forecast['ds'] = pd.to_datetime(sku_forecast['ds'])
    total_forecast = sku_forecast['pred_qty'].sum()
    
    # 📊 Ключевые метрики
    st.subheader("📊 Ключевые метрики для SKU")
    col1, col2 = st.columns(2)
    col1.metric("Общий прогноз на период (шт.)", f"{total_forecast:.0f}")
    col2.metric("Среднедневной прогноз (шт.)", f"{total_forecast / FORECAST_DAYS:.1f}")
    
    # 📈 График исторических + прогноз
    st.subheader("📈 Исторические продажи и прогноз")
    historical_sku = df[df['SKU'] == selected_sku][['ds', 'qty']].copy()
    historical_sku['ds'] = pd.to_datetime(historical_sku['ds'])
    historical_sku['type'] = 'Исторические'
    forecast_plot = sku_forecast.copy()
    forecast_plot['type'] = 'Прогноз'
    plot_df = pd.concat([historical_sku, forecast_plot])
    fig = px.line(plot_df, x='ds', y='qty' if 'qty' in plot_df.columns else 'pred_qty', color='type',
                  title=f"Продажи для {selected_item}", markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # 📊 Зависимость продаж от цены (историческая)
    st.subheader("📊 Зависимость продаж от цены")
    price_qty_df = df[df['SKU'] == selected_sku][['price', 'qty']].copy()
    fig_scatter = px.scatter(price_qty_df, x='price', y='qty', trendline="ols",
                             title=f"Зависимость для {selected_item}")
    fig_scatter.add_vline(x=promo_price, line_dash="dash", line_color="red",
                          annotation_text=f"Promo-цена: {promo_price:.1f} ₽")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 📋 Детали прогноза для SKU
    st.subheader("📋 Детали прогноза")
    details_df = pd.DataFrame({
        'Дата': sku_forecast['ds'].dt.strftime('%Y-%m-%d'),
        'Прогноз (шт.)': sku_forecast['pred_qty'].round(1)
    })
    st.dataframe(details_df)
    
    # Пояснения
    st.subheader("ℹ️ Пояснения")
    st.markdown("""
    - **Модель**: LightGBM с KFold ({N_FOLDS} фолдов), фичи: лаги, роллинги, временные, относительная цена.
    - **Прогноз**: На основе промо-цен из листа 'Цены', с клиппингом >=0.
    - **Метрики**: OOF RMSE — ошибка на валидации.
    """.format(N_FOLDS=N_FOLDS))
    
    # Экспорт
    if st.button("📥 Скачать полный прогноз как CSV"):
        full_pred = pred_df[['ds', 'SKU', 'pred_qty']].copy()
        full_pred['ds'] = pd.to_datetime(full_pred['ds'])
        full_pred.to_csv('promo_forecast.csv', index=False)
        st.download_button("Скачать", data=open('promo_forecast.csv', 'rb'), file_name='promo_forecast.csv')
    
    if st.button("💾 Сохранить полный прогноз в Excel"):
        pred_df[['ds', 'SKU', 'pred_qty']].to_excel('promo_forecast.xlsx', index=False)
        st.success("Файл сохранён: promo_forecast.xlsx")

else:
    st.info("Файл 'Продажи1.xlsx' загружен автоматически. Если нужно обновить, добавьте в репозиторий и передеплойте.")