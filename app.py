from flask import Flask, render_template, request
import os

from reg_analysis import run_boston_analysis
from tsa_analysis import run_power_analysis

app = Flask(__name__)

@app.route('/')
def dashboard():
    """
    메인 페이지 랜딩 화면을 렌더링합니다.
    """
    return render_template('index.html')

@app.route('/reg')
def regression_analysis():
    """
    보스턴 집값 회귀분석 페이지
    """
    boston_stats, boston_plot = None, None
    error_msg = None

    try:
        boston_stats, boston_plot = run_boston_analysis()
    except FileNotFoundError as e:
        error_msg = str(e)
    except Exception as e:
        error_msg = f"분석 연산 중 오류가 발생했습니다: {str(e)}"

    return render_template(
        'reg.html',
        boston_stats=boston_stats,
        boston_plot=boston_plot,
        error_msg=error_msg
    )

@app.route('/tsa')
def time_series_analysis():
    """
    전력 소비량 시계열 예측 페이지
    """
    power_forecast, power_plot = None, None
    error_msg = None

    try:
        power_forecast, power_plot = run_power_analysis()
    except FileNotFoundError as e:
        error_msg = str(e)
    except Exception as e:
        error_msg = f"분석 연산 중 오류가 발생했습니다: {str(e)}"

    return render_template(
        'tsa.html',
        power_forecast=power_forecast,
        power_plot=power_plot,
        error_msg=error_msg
    )

if __name__ == '__main__':
    # 디버그 모드로 구동하여 편리한 테스트 지원
    app.run(debug=True, host='127.0.0.1', port=5000)
