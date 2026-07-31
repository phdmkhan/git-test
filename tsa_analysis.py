import io
import base64
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from functools import lru_cache

# matplotlib GUI 렌더링 방지
import matplotlib
matplotlib.use('Agg')

DATA_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'Holt_Winters.txt')

def perform_backtesting(df):
    """
    마지막 12개월 데이터를 분리하여 MAPE와 RMSE 모델 성능을 평가합니다 (PRD 7.2 참고)
    """
    if len(df) <= 12:
        return None, None
    
    train_df = df.iloc[:-12]
    test_df = df.iloc[-12:]
    
    model = ExponentialSmoothing(train_df['MW'], trend=None, seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()
    predictions = fitted_model.forecast(12)
    
    mape = mean_absolute_percentage_error(test_df['MW'], predictions)
    rmse = np.sqrt(mean_squared_error(test_df['MW'], predictions))
    
    print("=== Holt-Winters 모델 백테스팅 검증 결과 ===")
    print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")
    print(f"RMSE: {rmse:.2f}")
    if mape < 0.10:
        print("-> 판정 결과: 모델 성능 '우수'")
    else:
        print("-> 판정 결과: 모델 편차 검토 필요")
    print("=========================================\n")


@lru_cache(maxsize=1)
def run_power_analysis():
    """
    전력 데이터를 통해 향후 12개월을 예측하고 결과 테이블용 리스트와 시각화 바이너리를 반환합니다.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"전력 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

    # 1. 데이터 로드 및 전처리
    df = pd.read_csv(DATA_PATH, sep=r'\s+')
    # '201201' 형태를 '2012-01-01' 등의 시계열로 변환
    df['Month'] = pd.to_datetime(df['Month'].astype(str), format='%Y%m')
    df.set_index('Month', inplace=True)
    df.index.freq = pd.infer_freq(df.index) or 'MS'

    # 2. 모델 검증 수행 (백테스팅)
    perform_backtesting(df)

    # 3. 전체 데이터로 실 예측 모델 훈련
    model = ExponentialSmoothing(df['MW'], trend=None, seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()
    
    # 4. 향후 12개월(1년) 수치 예측
    forecast = fitted_model.forecast(12)
    
    # 결과 테이블 형태 정제: [{'month': 'YYYY-MM', 'pred': 12345.67}, ...]
    power_forecast = [
        {
            'month': date.strftime('%Y-%m'), 
            'pred': round(float(val), 2)
        } 
        for date, val in zip(forecast.index, forecast.values)
    ]

    # 5. 시각화 (과거 데이터 + 미래 예측치 결합)
    plt.figure(figsize=(8, 6))
    plt.plot(df.index, df['MW'], label='Historical (MW)', color='royalblue')
    plt.plot(forecast.index, forecast.values, label='Forecast (1yr)', linestyle='--', color='orange')
    plt.title('Power Consumption: Historical & 1-Year Forecast')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (MW)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    power_plot = f"data:image/png;base64,{img_base64}"

    return power_forecast, power_plot

if __name__ == '__main__':
    forecast, plot = run_power_analysis()
    print("Power forecast:", forecast[:2], "...")
