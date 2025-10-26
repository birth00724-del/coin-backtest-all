import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# SuperTrend 계산 함수
# -------------------------------
def supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    df['ATR'] = df['High'].combine(df['Low'], np.subtract).abs().rolling(period).mean()
    df['UpperBand'] = hl2 + (multiplier * df['ATR'])
    df['LowerBand'] = hl2 - (multiplier * df['ATR'])

    trend = [True]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['UpperBand'].iloc[i - 1]:
            trend.append(True)
        elif df['Close'].iloc[i] < df['LowerBand'].iloc[i - 1]:
            trend.append(False)
        else:
            trend.append(trend[-1])
            if trend[-1] and df['LowerBand'].iloc[i] < df['LowerBand'].iloc[i - 1]:
                df.loc[df.index[i], 'LowerBand'] = df['LowerBand'].iloc[i - 1]
            if not trend[-1] and df['UpperBand'].iloc[i] > df['UpperBand'].iloc[i - 1]:
                df.loc[df.index[i], 'UpperBand'] = df['UpperBand'].iloc[i - 1]
    df['Supertrend'] = np.where(trend, df['LowerBand'], df['UpperBand'])
    df['Direction'] = np.where(df['Close'] >= df['Supertrend'], 1, -1)
    return df

# -------------------------------
# 백테스트 함수
# -------------------------------
def backtest(df, st_params, slippage=0.001, initial_capital=1000000):
    st1 = supertrend(df.copy(), period=st_params[0][0], multiplier=st_params[0][1])
    st2 = supertrend(df.copy(), period=st_params[1][0], multiplier=st_params[1][1])
    st3 = supertrend(df.copy(), period=st_params[2][0], multiplier=st_params[2][1])

    dir_df = pd.concat([st1['Direction'], st2['Direction'], st3['Direction']], axis=1)
    dir_df.columns = ['ST1', 'ST2', 'ST3']

    combined_signal = np.where(dir_df.sum(axis=1) == 3, 1, np.where(dir_df.sum(axis=1) == -3, -1, 0))
    df['Signal'] = pd.Series(combined_signal, index=df.index)

    # 매매 기록용 리스트
    trades = []
    position = 0
    entry_price = 0
    capital = initial_capital
    equity_curve = []

    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i]
        close = df['Close'].iloc[i]

        if signal == 1 and position == 0:  # 진입
            entry_price = close * (1 + slippage)
            position = 1
            entry_date = df.index[i]
        elif signal == -1 and position == 1:  # 청산
            exit_price = close * (1 - slippage)
            exit_date = df.index[i]
            ret = (exit_price - entry_price) / entry_price
            capital *= (1 + ret)
            trades.append({
                '매수일': entry_date.strftime('%Y-%m-%d'),
                '매수가': round(entry_price, 2),
                '매도일': exit_date.strftime('%Y-%m-%d'),
                '매도가': round(exit_price, 2),
                '슬리피지 반영 수익률(%)': round(ret * 100, 2),
                '자본 변화(원)': round(capital, 2)
            })
            position = 0
        equity_curve.append(capital)

    df['Equity'] = equity_curve + [capital] * (len(df) - len(equity_curve))
    df['Equity'] = df['Equity'].ffill()

    total_return = df['Equity'].iloc[-1] / initial_capital - 1
    years = (df.index[-1] - df.index[0]).days / 365
    CAGR = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    drawdown = (df['Equity'] / df['Equity'].cummax() - 1).min()
    returns = pd.Series(df['Equity']).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty else 0

    trade_df = pd.DataFrame(trades)
    return df, trade_df, CAGR, drawdown, sharpe

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 SuperTrend 3중 결합 백테스트")

uploaded_file = st.file_uploader("CSV 파일 업로드 (열: Date, Open, High, Low, Close, Volume)", type=['csv'])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.sort_index()

    st.sidebar.header("SuperTrend 파라미터 설정")
    st1 = (st.sidebar.number_input("ST1 Period", 5, 50, 10),
           st.sidebar.number_input("ST1 Multiplier", 1.0, 10.0, 3.0))
    st2 = (st.sidebar.number_input("ST2 Period", 5, 50, 15),
           st.sidebar.number_input("ST2 Multiplier", 1.0, 10.0, 4.0))
    st3 = (st.sidebar.number_input("ST3 Period", 5, 50, 20),
           st.sidebar.number_input("ST3 Multiplier", 1.0, 10.0, 5.0))

    slippage = st.sidebar.number_input("Slippage (예: 0.001 = 0.1%)", 0.0, 0.01, 0.001)
    initial_capital = st.sidebar.number_input("초기자금 (원)", 100000, 100000000, 1000000)

    if st.button("백테스트 실행"):
        df_result, trade_log, CAGR, MDD, Sharpe = backtest(data, [st1, st2, st3], slippage, initial_capital)

        st.subheader("📈 백테스트 결과 요약")
        st.write(f"**CAGR:** {CAGR*100:.2f}%")
        st.write(f"**MDD:** {MDD*100:.2f}%")
        st.write(f"**Sharpe Ratio:** {Sharpe:.2f}")
        st.write(f"**총 거래 횟수:** {len(trade_log)}회")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Equity'], name='Equity', line=dict(color='blue')))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📜 매매 내역")
        st.dataframe(trade_log)

        csv = trade_log.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 매매내역 다운로드 (CSV)", csv, "trade_log.csv", "text/csv")
