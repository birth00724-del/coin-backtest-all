import streamlit as st
import pandas as pd
import numpy as np

st.title("📈 Supertrend 3중 조합 백테스트 (1개라도 반대면 청산)")

# ====== Supertrend 함수 ======
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

        # 밴드 조정
        if df['Supertrend'][i]:
            df.loc[df.index[i], 'LowerBand'] = max(df['LowerBand'][i], df['LowerBand'][i - 1])
        else:
            df.loc[df.index[i], 'UpperBand'] = min(df['UpperBand'][i], df['UpperBand'][i - 1])
    return df

# ====== 백테스트 함수 ======
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

    for i in range(len(df)):
        price = df['Close'][i]

        # 매수 조건
        if df['BuySignal'][i] and position == 0:
            buy_price = price * (1 + slippage)
            position = capital / buy_price
            capital = 0
            buy_date = df.index[i]

        # 청산 조건 (1개라도 반대)
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

    # 마지막 포지션 정리
    if position > 0:
        capital = position * df['Close'].iloc[-1]
        position = 0

    total_return = capital / initial_capital
    years = (df.index[-1] - df.index[0]).days / 365.25
    CAGR = (total_return ** (1 / years) - 1) * 100 if years > 0 else np.nan

    # MDD 계산
    df['equity'] = np.maximum.accumulate(df['Close'] / df['Close'].iloc[0] * initial_capital)
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1
    MDD = df['drawdown'].min() * 100

    # Sharpe Ratio 계산
    df['return'] = df['Close'].pct_change()
    Sharpe = (df['return'].mean() / df['return'].std()) * np.sqrt(252)

    trade_log_df = pd.DataFrame(trade_log)
    return trade_log_df, CAGR, MDD, Sharpe

# ====== Streamlit 인터페이스 ======
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # ✅ 컬럼명 정규화
    data.columns = [col.strip().capitalize() for col in data.columns]
    rename_map = {
        '날짜': 'Date', '시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'
    }
    data.rename(columns=rename_map, inplace=True)

    # ✅ 필요한 컬럼 확인
    required = ['Date', 'Open', 'High', 'Low', 'Close']
    missing = [c for c in required if c not in data.columns]
    if missing:
        st.error(f"CSV에 다음 컬럼이 없습니다: {', '.join(missing)}")
        st.stop()

    # ✅ Date 처리
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    st.success("✅ 데이터 불러오기 완료")

    # ===== Supertrend 파라미터 =====
    st.subheader("Supertrend 설정")
    col1, col2, col3 = st.columns(3)
    with col1:
        st1_p = st.number_input("ST1 기간", 5, 50, 10)
        st1_m = st.number_input("ST1 배수", 1.0, 10.0, 3.0)
    with col2:
        st2_p = st.number_input("ST2 기간", 5, 50, 20)
        st2_m = st.number_input("ST2 배수", 1.0, 10.0, 4.0)
    with col3:
        st3_p = st.number_input("ST3 기간", 5, 50, 30)
        st3_m = st.number_input("ST3 배수", 1.0, 10.0, 5.0)

    slippage = st.number_input("슬리피지 비율", 0.0, 0.01, 0.001, format="%.4f")
    initial_capital = st.number_input("초기자금", 100000, 10000000, 1000000, step=100000)

    if st.button("🚀 백테스트 실행"):
        with st.spinner("계산 중..."):
            trades, CAGR, MDD, Sharpe = backtest(
                data, [(st1_p, st1_m), (st2_p, st2_m), (st3_p, st3_m)],
                slippage, initial_capital
            )

        st.subheader("📊 결과 요약")
        st.write(f"**CAGR:** {CAGR:.2f}%")
        st.write(f"**MDD:** {MDD:.2f}%")
        st.write(f"**Sharpe Ratio:** {Sharpe:.2f}")

        st.subheader("🧾 매매 내역")
        st.dataframe(trades)

        if not trades.empty:
            csv = trades.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="💾 매매 내역 다운로드",
                data=csv,
                file_name="trade_log.csv",
                mime="text/csv"
            )
