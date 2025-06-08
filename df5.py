import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from fredapi import Fred
import matplotlib.pyplot as plt
qt_plugin_path = r"C:\Users\Mike\AppData\Local\Programs\Python\Python310\Lib\site-packages\PyQt5\Qt5\plugins"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
# Инициализация FRED API (нужен ваш API ключ)
fred = Fred(api_key='8d726355d6964bb1ff17ede756a6d773')

def get_fred_data(series1_id, series2_id, start_date):
    """Загружает данные из FRED и объединяет их в DataFrame"""
    # Получаем данные и информацию о датах
    series1 = fred.get_series(series1_id)
    series2 = fred.get_series(series2_id)
    
    # Проверяем доступный диапазон дат
    min_date = max(series1.index.min(), series2.index.min()).strftime('%Y-%m-%d')
    max_date = min(series1.index.max(), series2.index.max()).strftime('%Y-%m-%d')
    
    print(f"\nДоступный диапазон дат для выбранных показателей:")
    print(f"Начальная дата: {min_date}")
    print(f"Конечная дата: {max_date}")
    
    # Фильтруем по выбранной дате
    series1 = series1[series1.index >= start_date]
    series2 = series2[series2.index >= start_date]
    
    # Приводим к месячной частоте
    if series1_id == 'DCOILWTICO':
        series1 = series1.resample('ME').mean()
    if series2_id == 'DCOILWTICO':
        series2 = series2.resample('ME').mean()
    
    # Выравниваем индексы
    series1.index = series1.index.to_period('M').to_timestamp('M')
    series2.index = series2.index.to_period('M').to_timestamp('M')
    
    df = pd.DataFrame({series1_id: series1, series2_id: series2}).dropna()
    return df
def get_unit(series_id):
    """Безопасное получение единиц измерения по ID серии"""
    for ind in indicators.values():
        if ind[0] == series_id:
            return ind[2]
    return "единицы"  # Значение по умолчанию
def analyze_relationship(df, x_col, y_col):
    """Анализирует линейную зависимость с центрированием"""
    # Центрируем X (вычитаем среднее)
    x_centered = df[x_col] - df[x_col].mean()
    X = sm.add_constant(x_centered)  # Теперь intercept будет иметь смысл среднего Y
    y = df[y_col]
    
    model = sm.OLS(y, X).fit()
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    
    # График с центрированными данными
    plt.subplot(121)
    plt.scatter(x_centered, y, alpha=0.5)
    plt.plot(x_centered, model.predict(X), 'r-')
    plt.xlabel(f"{x_col} (центрированная)")
    plt.ylabel(y_col)
    plt.title(f'Центрированные данные\nR²={model.rsquared:.3f}')
    plt.grid(True)
    
    # График с исходными данными
    plt.subplot(122)
    plt.scatter(df[x_col], y, alpha=0.5)
    plt.plot(df[x_col], model.predict(sm.add_constant(df[x_col] - df[x_col].mean())), 'r-')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Исходные данные\nIntercept: {model.params.iloc[0]:.2f}, Slope: {model.params.iloc[1]:.2f}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model

def prediction_interface(model, df, x_col, y_col):
    """Интерфейс для построения предсказательных интервалов"""
    # Параметры модели (используем iloc для избежания FutureWarning)
    a = model.params.iloc[0]
    b = model.params.iloc[1]
    n = len(df)
    residuals = model.resid
    s = np.sqrt(np.sum(residuals**2) / (n - 2))
    x_mean = df[x_col].mean()
    Sxx = np.sum((df[x_col] - x_mean)**2)
    
    def get_prediction_interval(x0):
        x0_centered = x0 - x_mean
        y_pred = a + b * x0_centered
        se = s * np.sqrt(1 + 1/n + (x0_centered**2 / Sxx))
        t_crit = stats.t.ppf(0.975, df=n-2)
        return y_pred, (y_pred - t_crit * se, y_pred + t_crit * se)
    
    # Основной интерфейс
    print("\n" + "="*50)
    print(f"Анализ зависимости {y_col} от {x_col}")
    print(f"Доступный диапазон {x_col}: [{df[x_col].min():.2f}, {df[x_col].max():.2f}]")
    print(f"Среднее значение {x_col}: {x_mean:.2f}")
    print(f"Модель: {y_col} = {a:.4f} + {b:.4f}*(FEDFUNDS - {x_mean:.2f})")
    print(f"R²: {model.rsquared:.4f}")
    print("="*50 + "\n")
    
    while True:
        try:
            x_input = input(f"Введите значение {x_col} или 'q' для выхода: ")
            if x_input.lower() == 'q':
                break
            
            x0 = float(x_input)
            y_pred, pi = get_prediction_interval(x0)
            unit_x = get_unit(x_col)
            unit_y = get_unit(y_col)
            print(f"\nДля {x_col} = {x0:.2f} {unit_x}:")
            print(f"Предсказанное {y_col}: {y_pred:.2f} {unit_y}")
            print(f"95% Prediction Interval: [{pi[0]:.2f}, {pi[1]:.2f}] {unit_y}")
            
            if x0 < df[x_col].min() or x0 > df[x_col].max():
                print("⚠️ Внимание: экстраполяция за пределы данных!")
            
            print("\n" + "="*50 + "\n")
            
        except ValueError:
            print("Ошибка: введите число или 'q' для выхода")

if __name__ == "__main__":
    indicators = {
        '1': ('DCOILWTICO', 'Нефть (WTI)', 'долларов за баррель'),
        '2': ('CPIAUCSL', 'Инфляция (CPI)', 'индекс (1982-1984=100)'),
        '3': ('UNRATE', 'Уровень безработицы', '%'),
        '4': ('GDPC1', 'ВВП (реальный)', 'млрд долларов 2012 года'),
        '5': ('FEDFUNDS', 'Ставка ФРС', '%'),
        '6': ('SP500', 'Индекс S&P 500', 'пунктов')
    }
    
    print("Доступные экономические показатели:")
    for key, value in indicators.items():
        id_, name, unit = value  # Распаковываем кортеж из 3 элементов
        print(f"{key}: {name} ({id_}) - единицы: {unit}")
    
    x_id = input("\nВведите номер показателя для X (независимая переменная): ")
    y_id = input("Введите номер показателя для Y (зависимая переменная): ")
    
    # Получаем информацию о доступных датах перед запросом
    x_series = indicators[x_id][0]
    y_series = indicators[y_id][0]
    x_data = fred.get_series(x_series)
    y_data = fred.get_series(y_series)
    min_date = max(x_data.index.min(), y_data.index.min()).strftime('%Y-%m-%d')
    max_date = min(x_data.index.max(), y_data.index.max()).strftime('%Y-%m-%d')
    
    print(f"\nДоступный диапазон дат: с {min_date} по {max_date}")
    start_date = input("Введите начальную дату (формат YYYY-MM-DD): ")
    
    try:
        df = get_fred_data(x_series, y_series, start_date)
        print("\nПервые 5 строк данных:")
        print(df.head())
        
        model = analyze_relationship(df, x_series, y_series)
        prediction_interface(model, df, x_series, y_series)
        
    except KeyError:
        print("Ошибка: неверный номер показателя")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")