import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("📈 Supertrend 3중 조합 백테스트 (1개라도 반대면 청산 + 자산 그래프)")

# ===== Supertrend 함수 =====
def supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(period).mean()
    df['UpperBand'] = hl2 + (multiplier * df['ATR'])
    df['LowerBand'] = hl2 - (multiplier * df['ATR'])
    df['Supertrend'] = True

    for i in range(1, len(df)):
        if df['Close'][i] > df['UpperBand'][i - 1]:
            df.loc[df.index[i], 'Supertrend'] = True
        elif df['Close'][i] < df['LowerBand'][i - 1]:
            df.loc[df.index[i], 'Supertrend'] = False
        else:
            df.loc[df.index[i], 'Supertrend'] = df['Supertrend'][i - 1]

        if df['Supertrend'][i]:
            df.loc[df.index[i], 'LowerBand'] = max(df['LowerBand'][i], df['LowerBand'][i - 1])
        else:
            df.loc[df.index[i], 'UpperBand'] = min(df['UpperBand'][i], df['UpperBand'][i - 1])
    return df

# ===== 백테스트 함수 =====
def backtest(df, st_params, slippage=0.001, initial_capital=1000000):
    st1 = supertrend(df.copy(), period=st_params[0][0], multiplier=st_params[0][1])
    st2 = supertrend(df.copy(), period=st_params[1][0], multiplier=st_params[1][1])
    st3 = supertrend(df.copy(), period=st_params[2][0], multiplier=st_params[2][1])

    # 매수: 3개 모두 상승 / 매도: 1개라도 하락
    df['BuySignal'] = (st1['Supertrend'] & st2['Supertrend'] & st3['Supertrend'])
    df['SellSignal'] = (~st1['Supertrend']) | (~st2['Supertrend']) | (~st3['Supertrend'])

    position = 0
    buy_price = 0
    capital = initial_capital
    trade_log = []
    equity_curve = []

    for i in range(len(df)):
        price = df['Close'][i]
        # 매수
        if df['BuySignal'][i] and position == 0:
            buy_price = price * (1 + slippage)
            position = capital / buy_price
            capital = 0
            buy_date = df.index[i]
        # 청산
        elif df['SellSignal'][i] and position > 0:
            sell_price = price * (1 - slippage)
            capital = position * sell_price
            position = 0
            sell_date = df.index[i]
            ret = (sell_price - buy_price) / buy_price
            trade_log.append({
                '매수일': buy_date.date(),
                '매수가': round(buy_price, 2),
                '매도일': sell_date.date(),
                '매도가': round(sell_price, 2),
                '수익률(%)': round(ret * 100, 2),
                '자산': round(capital, 2)
            })
        equity_curve.append(capital if capital > 0 else position * price)

    df['Equity'] = equity_curve
    total_return = df['Equity'].iloc[-1] / initial_capital
    years = (df.index[-1] - df.index[0]).days / 365.25
    CAGR = (total_return ** (1 / years) - 1) * 100 if years > 0 else np.nan
    MDD = ((df['Equity'] / df['Equity'].cummax()) - 1).min() * 100
    returns = df['Equity'].pct_change().dropna()
    Sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty else 0

    trade_log_df = pd.DataFrame(trade_log)
    return df, trade_log_df, CAGR, MDD, Sharpe

# ===== Streamlit UI =====
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # ✅ 컬럼명 정규화
    data.columns = [col.strip().capitalize() for col in data.columns]
    rename_map = {
        '날짜': 'Date', '시가': 'Open', '고가': 'High',
        '저가': 'Low', '종가': 'Close', '거래량': 'Volume'
    }
    data.rename(columns=rename_map, inplace=True)

    # ✅ 날짜 컬럼 자동 인식
    date_candidates = [c for c in data.columns if c.lower() in ['date', 'datetime', 'timestamp', '날짜']]
    if not date_candidates:
        st.error("❌ CSV에서 날짜 컬럼(Date, datetime, timestamp, 날짜 등)을 찾지 못했습니다.")
        st.write("현재 CSV 컬럼명:", list(data.columns))
        st.stop()

    data[date_candidates[0]] = pd.to_datetime(data[date_candidates[0]])
    data.set_index(date_candidates[0], inplace=True)

    # ✅ 가격 컬럼 확인
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        st.error(f"CSV에 다음 컬럼이 없습니다: {', '.join(missing)}")
        st.stop()

    st.success("✅ 데이터 불러오기 완료")

    # ===== 파라미터 입력 =====
    st.sidebar.header("Supertrend 파라미터 설정")
    st1 = (st.sidebar.number_input("ST1 기간", 5, 50, 10), st.sidebar.number_input("ST1 배수", 1.0, 10.0, 3.0))
    st2 = (st.sidebar.number_input("ST2 기간", 5, 50, 20), st.sidebar.number_input("ST2 배수", 1.0, 10.0, 4.0))
    st3 = (st.sidebar.number_input("ST3 기간", 5, 50, 30), st.sidebar.number_input("ST3 배수", 1.0, 10.0, 5.0))
    slippage = st.sidebar.number_input("슬리피지 비율", 0.0, 0.01, 0.001, format="%.4f")
    initial_capital = st.sidebar.number_input("초기자금 (원)", 100000, 100000000, 1000000, step=100000)

    # ===== 실행 버튼 =====
    if st.button("🚀 백테스트 실행"):
        with st.spinner("계산 중..."):
            df_result, trade_log, CAGR, MDD, Sharpe = backtest(
                data, [st1, st2, st3], slippage, initial_capital
            )

        st.subheader("📊 결과 요약")
        st.write(f"**CAGR:** {CAGR:.2f}%")
        st.write(f"**MDD:** {MDD:.2f}%")
        st.write(f"**Sharpe Ratio:** {Sharpe:.2f}")
        st.write(f"**총 거래 횟수:** {len(trade_log)}회")

        # ===== 그래프 시각화 =====
        st.subheader("📈 자산 곡선 (Equity Curve)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Equity'], mode='lines', name='Equity', line=dict(color='blue')))
        fig.update_layout(xaxis_title="날짜", yaxis_title="자산 (원)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # ===== 매매 내역 =====
        st.subheader("🧾 매매 내역")
        st.dataframe(trade_log)

        if not trade_log.empty:
            csv = trade_log.to_csv(index=False).encode('utf-8-sig')
            st.download_button("💾 매매 내역 다운로드", csv, "trade_log.csv", "text/csv")
