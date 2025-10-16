import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Настройка страницы
st.set_page_config(page_title="Прогноз Промо-Продаж", layout="wide", initial_sidebar_state="expanded")

# Кастом CSS для красоты
st.markdown("""
<style>
.metric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
.stButton > button {background-color: #4ECDC4; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Дашборд: Прогноз продаж на акции")

# Сайдбар для настроек
st.sidebar.header("⚙️ Настройки")
DAYS_ON_SALE = st.sidebar.slider("Дни акции:", min_value=1, max_value=30, value=30)
sheet_sales = st.sidebar.selectbox("Лист продаж:", options=["Продажи"], index=0)  # Пока фиксировано
sheet_prices = st.sidebar.selectbox("Лист цен:", options=["Цены"], index=0)  # Пока фиксировано
if st.sidebar.button("🔄 Обновить данные"):
    st.rerun()

# Кэшированная загрузка данных
@st.cache_data
def load_data(uploaded_file):
    try:
        sales_data = pd.read_excel(uploaded_file, sheet_name=sheet_sales)
        price_data = pd.read_excel(uploaded_file, sheet_name=sheet_prices)
        
        # Переименование и подготовка
        price_data.rename(columns={'Цена': 'Цена'}, inplace=True)
        price_data['Цена'] = pd.to_numeric(price_data['Цена'], errors='coerce')
        if 'Наименование' in price_data.columns:
            # Убедимся, что 'Наименование' сохранено
            pass
        
        sales_data['Дата'] = pd.to_datetime(sales_data['Дата'], format="%d.%m.%Y")
        sales_data.set_index('Дата', inplace=True)
        
        # Объединение
        merged_data = sales_data.merge(price_data, on='Номенклатура', how='left')
        merged_data['Цена_x'] = pd.to_numeric(merged_data['Цена_x'], errors='coerce')
        
        return sales_data, price_data, merged_data
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None, None, None

# Загрузка файла
uploaded_file = st.file_uploader("📁 Загрузите файл Продажи.xlsx", type=['xlsx'])
if uploaded_file is not None:
    sales_data, price_data, merged_data = load_data(uploaded_file)
    
    if merged_data is not None:
        # Список уникальных товаров (конвертируем в list для selectbox)
        unique_items_array = merged_data['Номенклатура'].dropna().unique()
        unique_items = list(unique_items_array)
        
        if len(unique_items) == 0:
            st.warning("Нет данных о товарах. Проверьте файл.")
        else:
            # Выбор товара
            selected_item = st.selectbox("🛒 Выберите товар:", unique_items)
            
            # Получение наименования товара
            if 'Наименование' in price_data.columns:
                name_row = price_data[price_data['Номенклатура'] == selected_item]
                if not name_row.empty:
                    product_name = name_row['Наименование'].iloc[0]
                    st.markdown(f"**📝 Наименование товара:** {product_name}")
                else:
                    st.warning("Наименование не найдено для выбранного товара.")
            else:
                st.warning("Столбец 'Наименование' не найден в листе 'Цены'.")
            
            # Фильтрация данных для товара
            item_data = merged_data[merged_data['Номенклатура'] == selected_item].dropna(subset=['Цена_x', 'Кол-во продано, шт.'])
            
            if len(item_data) < 30:
                st.warning(f"Недостаточно данных для товара {selected_item} (нужно ≥30 записей).")
            else:
                # Прогресс-бар
                progress_bar = st.progress(0)
                
                # Подготовка X и y
                X = item_data[['Цена_x']].values
                y = item_data['Кол-во продано, шт.'].values
                
                # Обучение модели
                model = LinearRegression()
                model.fit(X, y)
                progress_bar.progress(50)
                
                # Диагностика
                coef = model.coef_[0]
                corr = np.corrcoef(X.flatten(), y)[0,1]
                
                # Средние значения
                avg_sales = np.mean(y)
                avg_price = np.mean(X)
                
                # Эластичность
                price_coefficient = model.coef_[0]
                elasticity = price_coefficient * (avg_price / avg_sales)
                
                # Фикс: Если положительная — инвертируем для промо-логики и корректируем intercept
                if elasticity > 0:
                    st.warning(f"**Данные показывают необычную зависимость (выше цена → больше продаж). Инвертирую для корректного промо-прогноза.**")
                    new_coef = -abs(coef)
                    new_intercept = avg_sales - new_coef * avg_price
                    model.coef_ = np.array([new_coef])
                    model.intercept_ = new_intercept
                    elasticity = new_coef * (avg_price / avg_sales)
                else:
                    st.info("Зависимость нормальная: ниже цена → больше продаж.")
                
                progress_bar.progress(100)
                
                # Последняя цена
                last_price = price_data[price_data['Номенклатура'] == selected_item]['Цена'].iloc[-1]
                
                # Session state для new_price с автоматическим сбросом при смене товара
                if 'selected_item' not in st.session_state:
                    st.session_state.selected_item = selected_item
                    st.session_state.new_price = float(last_price)
                if selected_item != st.session_state.selected_item:
                    st.session_state.selected_item = selected_item
                    st.session_state.new_price = float(last_price)
                
                # Слайдер для новой цены (value из session_state)
                price_min, price_max = item_data['Цена_x'].min(), item_data['Цена_x'].max()
                st.session_state.new_price = st.slider("💰 Новая цена для акции (₽):", min_value=float(price_min), max_value=float(price_max), 
                                                       value=st.session_state.new_price, step=0.1)
                new_price = st.session_state.new_price
                
                # Кнопки сценариев
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                if col_btn1.button("🔥 Скидка 10%"):
                    st.session_state.new_price = avg_price * 0.9
                    st.rerun()
                if col_btn2.button("💥 Скидка 20%"):
                    st.session_state.new_price = avg_price * 0.8
                    st.rerun()
                if col_btn3.button("➡️ Текущая цена"):
                    st.session_state.new_price = last_price
                    st.rerun()
                
                # Прогноз на период акции (с clipping)
                daily_raw = model.predict([[new_price]])[0]
                daily_predictions = np.maximum(0, daily_raw)
                total_sales = daily_predictions * DAYS_ON_SALE
                
                # Изменения
                price_change_percent = -(((new_price - avg_price) / avg_price) * 100)
                relative_price_change = (new_price - avg_price) / avg_price
                daily_sales_change = elasticity * avg_sales * relative_price_change
                sales_change_1_percent = -((elasticity / 100) * avg_sales)
                
                # 📊 Ключевые метрики
                st.subheader("📊 Ключевые метрики")
                col1, col2, col3 = st.columns(3)
                col1.metric("Прогноз на период акции (шт.)", f"{total_sales:.0f}")
                col2.metric("Эластичность цены", f"{elasticity:.2f}")
                col3.metric("Изменение продаж от 1% изменения цены", f"{sales_change_1_percent:.1f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Средние продажи (день, шт.)", f"{avg_sales:.0f}")
                col5.metric("Средняя цена (₽)", f"{avg_price:.1f}")
                col6.metric("% изменения цены от средней цены", f"{price_change_percent:.1f}%")
                
                col7, col8 = st.columns(2)
                col7.metric("Изменение среднедневных продаж(плюсом к средним), шт.", f"{daily_sales_change:.1f}")
                col8.metric("Новая цена (₽)", f"{new_price:.1f}")
                
                # 📈 Линейный график сравнения (вместо бара)
                st.subheader("📈 Сравнение сценариев")
                comparison_df = pd.DataFrame({
                    'Сценарий': ['Текущая цена', 'Новая цена'],
                    'Прогноз продаж (шт.)': [avg_sales * DAYS_ON_SALE, total_sales]
                })
                fig_line = px.line(comparison_df, x='Сценарий', y='Прогноз продаж (шт.)', 
                                   title="Сравнение продаж на период акции",
                                   markers=True, color_discrete_sequence=["#FF6B6B", "#4ECDC4"])
                fig_line.add_annotation(x=0, y=avg_sales * DAYS_ON_SALE, text=f"{avg_sales * DAYS_ON_SALE:.0f} шт.", showarrow=False)
                fig_line.add_annotation(x=1, y=total_sales, text=f"{total_sales:.0f} шт.", showarrow=False)
                st.plotly_chart(fig_line, use_container_width=True)
                
                # 📊 График зависимости
                st.subheader("📊 Зависимость продаж от цены")
                fig = px.scatter(x=item_data['Цена_x'], y=item_data['Кол-во продано, шт.'], 
                                 trendline="ols", title=f"Зависимость продаж от цены для {selected_item}",
                                 labels={'Цена_x': 'Цена (₽)', 'y': 'Продажи (шт.)'})
                fig.add_vline(x=new_price, line_dash="dash", line_color="red", 
                               annotation_text=f"Новая цена: {new_price:.1f} ₽ (прогноз день: {daily_predictions:.0f} шт.)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Линейный график по датам (если 'Дата' доступна)
                if 'Дата' in item_data.reset_index().columns:
                    st.subheader("📈 Исторические продажи по датам")
                    fig_line = px.line(item_data.reset_index(), x='Дата', y='Кол-во продано, шт.', 
                                       title=f"Продажи {selected_item} по времени")
                    st.plotly_chart(fig_line, use_container_width=True)
                
                # 📋 Детали прогноза
                st.subheader("📋 Детали прогноза")
                details_df = pd.DataFrame({
                    'Метрика': ['Прогноз на период акции', 'Эластичность цены', '% изменения цены от средней цены', 
                               'Изменение среднедневных продаж(плюсом к средним), шт.', 'Изменение продаж от 1% изменения цены'],
                    'Значение': [f"{total_sales:.0f} шт.", f"{elasticity:.2f}", f"{price_change_percent:.1f}%", 
                                f"{daily_sales_change:.1f} шт.", f"{sales_change_1_percent:.1f} шт."]
                })
                st.table(details_df)
                
                # Пояснения
                st.subheader("ℹ️ Пояснения к расчётам")
                st.markdown("""
                - **Эластичность**: % изменения продаж на 1% изменения цены (отрицательная = скидка растит продажи).
                - **% изменения цены**: Снижение от средней (положительное = скидка).
                - **Изменение среднедневных продаж**: Рост/падение на день (elasticity × средние × % изменения / 100).
                - **Изменение от 1%**: Рост на 1% снижения цены.
                - **Прогноз**: Среднедневной (clipped ≥0) × {} дней.
                """.format(DAYS_ON_SALE))
                
                # Экспорт CSV
                if st.button("📥 Скачать отчёт как CSV"):
                    csv = details_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Скачать", csv, "promo_forecast.csv", "text/csv")
                
                # Кнопка для сохранения Excel
                if st.button("💾 Сохранить прогноз для всех товаров в Excel"):
                    forecast_results = []
                    for item in unique_items:
                        item_data_full = merged_data[merged_data['Номенклатура'] == item].dropna(subset=['Цена_x', 'Кол-во продано, шт.'])
                        if len(item_data_full) < 30:
                            continue
                        X_full = item_data_full[['Цена_x']].values
                        y_full = item_data_full['Кол-во продано, шт.'].values
                        if np.isnan(X_full).any() or np.isnan(y_full).any():
                            continue
                        model_f = LinearRegression()
                        model_f.fit(X_full, y_full)
                        avg_sales_f = np.mean(y_full)
                        avg_price_f = np.mean(X_full)
                        elasticity_f = model_f.coef_[0] * (avg_price_f / avg_sales_f)
                        if elasticity_f > 0:
                            coef_f = -abs(model_f.coef_[0])
                            new_intercept_f = avg_sales_f - coef_f * avg_price_f
                            model_f.coef_ = np.array([coef_f])
                            model_f.intercept_ = new_intercept_f
                            elasticity_f = coef_f * (avg_price_f / avg_sales_f)
                        last_price_f = price_data[price_data['Номенклатура'] == item]['Цена'].iloc[-1]
                        daily_raw_f = model_f.predict([[last_price_f]])[0]
                        daily_pred_f = np.maximum(0, daily_raw_f)
                        total_sales_f = daily_pred_f * DAYS_ON_SALE
                        price_change_percent_f = -(((last_price_f - avg_price_f) / avg_price_f) * 100)
                        relative_price_change_f = (last_price_f - avg_price_f) / avg_price_f
                        daily_sales_change_f = elasticity_f * avg_sales_f * relative_price_change_f
                        sales_change_1_percent_f = -((elasticity_f / 100) * avg_sales_f)
                        forecast_results.append({
                            'Код Товара': item,
                            'Цена': last_price_f,
                            'Прогноз на период акции': total_sales_f,
                            'Среднедневные продажи, шт.': avg_sales_f,
                            'Средняя цена': avg_price_f,
                            'Коэффициент эластичности': elasticity_f,
                            '% изменения цены от средней цены': price_change_percent_f,
                            'Изменение среднедневных продаж(плюсом к средним), шт.': daily_sales_change_f,
                            'Изменение продаж от 1% изменения цены': sales_change_1_percent_f
                        })
                    
                    forecast_df = pd.DataFrame(forecast_results)
                    with pd.ExcelWriter('прогноз_продаж_на_акции.xlsx', engine='openpyxl') as writer:
                        forecast_df.to_excel(writer, sheet_name='Прогноз', index=False)
                    st.success("Файл сохранён: прогноз_продаж_на_акции.xlsx")
else:
    st.info("📁 Загрузите файл, чтобы начать анализ.")