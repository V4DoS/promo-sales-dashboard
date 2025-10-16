import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Глобальная переменная для дней акции
DAYS_ON_SALE = 30

# Настройка страницы
st.set_page_config(page_title="Прогноз Промо-Продаж", layout="wide")
st.title("Дашборд: Прогноз продаж на акции")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл Продажи.xlsx", type=['xlsx'])
if uploaded_file is not None:
    try:
        # Чтение данных
        sales_data = pd.read_excel(uploaded_file, sheet_name='Продажи')
        price_data = pd.read_excel(uploaded_file, sheet_name='Цены')
        
        # Переименование и подготовка (как в вашем коде)
        price_data.rename(columns={'Цена': 'Цена'}, inplace=True)
        price_data['Цена'] = pd.to_numeric(price_data['Цена'], errors='coerce')
        
        sales_data['Дата'] = pd.to_datetime(sales_data['Дата'], format="%d.%m.%Y")
        sales_data.set_index('Дата', inplace=True)
        
        # Объединение
        merged_data = sales_data.merge(price_data, on='Номенклатура', how='left')
        merged_data['Цена_x'] = pd.to_numeric(merged_data['Цена_x'], errors='coerce')
        
        # Список уникальных товаров
        unique_items = merged_data['Номенклатура'].dropna().unique()
        
        if len(unique_items) == 0:
            st.warning("Нет данных о товарах. Проверьте файл.")
        else:
            # Выбор товара
            selected_item = st.selectbox("Выберите товар:", unique_items)
            
            # Фильтрация данных для товара
            item_data = merged_data[merged_data['Номенклатура'] == selected_item].dropna(subset=['Цена_x', 'Кол-во продано, шт.'])
            
            if len(item_data) < 30:
                st.warning(f"Недостаточно данных для товара {selected_item} (нужно ≥30 записей).")
            else:
                # Подготовка X и y (все данные, без split)
                X = item_data[['Цена_x']].values
                y = item_data['Кол-во продано, шт.'].values
                
                # Обучение модели
                model = LinearRegression()
                model.fit(X, y)
                
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
                
                # Последняя цена
                last_price = price_data[price_data['Номенклатура'] == selected_item]['Цена'].iloc[-1]
                
                # Слайдер (default на последней цене)
                price_min, price_max = item_data['Цена_x'].min(), item_data['Цена_x'].max()
                new_price = st.slider("Новая цена для акции (₽):", min_value=float(price_min), max_value=float(price_max), 
                                      value=float(last_price), step=0.1)
                
                # Прогноз на период акции (с clipping для >=0)
                daily_raw = model.predict([[new_price]])[0]
                daily_predictions = np.maximum(0, daily_raw)  # Clipping
                total_sales = daily_predictions * DAYS_ON_SALE
                
                # Изменения (как в оригинале)
                price_change_percent = -(((new_price - avg_price) / avg_price) * 100)  # Положительное для скидки
                relative_price_change = (new_price - avg_price) / avg_price
                daily_sales_change = elasticity * avg_sales * relative_price_change
                sales_change_1_percent = -((elasticity / 100) * avg_sales)  # Положительное для снижения
                
                # Отображение метрик в колонках
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
                
                # График
                fig = px.scatter(x=item_data['Цена_x'], y=item_data['Кол-во продано, шт.'], 
                                 trendline="ols", title=f"Зависимость продаж от цены для {selected_item}",
                                 labels={'Цена_x': 'Цена (₽)', 'y': 'Продажи (шт.)'})
                fig.add_vline(x=new_price, line_dash="dash", line_color="red", 
                               annotation_text=f"Новая цена: {new_price:.1f} ₽ (прогноз день: {daily_predictions:.0f} шт.)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Таблица с деталями
                st.subheader("Детали прогноза")
                details_df = pd.DataFrame({
                    'Метрика': ['Прогноз на период акции', 'Эластичность цены', '% изменения цены от средней цены', 
                               'Изменение среднедневных продаж(плюсом к средним), шт.', 'Изменение продаж от 1% изменения цены'],
                    'Значение': [f"{total_sales:.0f} шт.", f"{elasticity:.2f}", f"{price_change_percent:.1f}%", 
                                f"{daily_sales_change:.1f} шт.", f"{sales_change_1_percent:.1f} шт."]
                })
                st.table(details_df)
                
                # Пояснения
                st.subheader("Пояснения к расчётам")
                st.markdown("""
                - **Эластичность**: % изменения продаж на 1% изменения цены (отрицательная = скидка растит продажи).
                - **% изменения цены**: Снижение от средней (положительное = скидка).
                - **Изменение среднедневных продаж**: Рост/падение на день (elasticity × средние × % изменения / 100).
                - **Изменение от 1%**: Рост на 1% снижения цены.
                - **Прогноз**: Среднедневной (clipped ≥0) × 5 дней.
                """)
                
                # Кнопка для сохранения (с фиксом intercept)
                if st.button("Сохранить прогноз для всех товаров в Excel"):
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
    except Exception as e:
        st.error(f"Ошибка в данных: {e}. Проверьте столбцы в Excel.")
else:
    st.info("Загрузите файл, чтобы начать анализ.")