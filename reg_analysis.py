import io
import base64
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache

# matplotlib GUI 렌더링 방지 (백그라운드 이미지 생성)
import matplotlib
matplotlib.use('Agg')

DATA_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'kaggle_boston_price.csv')

@lru_cache(maxsize=1)
def run_boston_analysis():
    """
    보스턴 집값 데이터를 로드하고 단순 선형 회귀 모형을 구축한 뒤
    평가지표와 그래프(Base64)를 반환하는 통합 함수입니다.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"보스턴 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

    # 1. 데이터 로드
    df = pd.read_csv(DATA_PATH)
    
    # 2. 독립변수(RM)와 종속변수(CMEDV) 설정
    X = df['RM']
    y = df['CMEDV']
    
    # statsmodels를 위한 상수항 추가
    X_sm = sm.add_constant(X)
    
    # 3. OLS 모델 적합
    model = sm.OLS(y, X_sm)
    results = model.fit()
    
    # 4. 통계 지표 산출
    # r2: 결정계수, coef: 회귀계수(RM), p_value: RM의 유의확률
    r2 = results.rsquared
    coef = results.params['RM']
    p_value = results.pvalues['RM']
    
    boston_stats = {
        'r2': round(r2, 4),
        'coef': round(coef, 4),
        'p_value': p_value
    }

    # 5. 시각화 그래프 생성 (Scatter Plot + Regression Line)
    plt.figure(figsize=(8, 6))
    sns.regplot(x='RM', y='CMEDV', data=df, 
                scatter_kws={'alpha': 0.5, 'color': 'steelblue'}, 
                line_kws={'color': 'red'})
    plt.title('Boston Housing: RM vs CMEDV')
    plt.xlabel('Average Number of Rooms (RM)')
    plt.ylabel('Corrected Median Value (CMEDV)')
    
    # 인메모리 버퍼에 저장하고 base64 추출
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    boston_plot = f"data:image/png;base64,{img_base64}"
    
    return boston_stats, boston_plot

if __name__ == '__main__':
    stats, plot = run_boston_analysis()
    print("Boston Stats:", stats)
    # print("Boston Plot length:", len(plot))
