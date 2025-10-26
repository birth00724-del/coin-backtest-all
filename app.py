import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Supertrend 3중(종가체결) 백테스터", layout="wide")
st.title("📈 Supertrend 3중 결합 — 종가 기준 매매 (date_utc)")

# ===============================
# Supertrend 계산 (Series 기반)
# ===============================
def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    d = df.copy()
    for c in ["High", "Low", "Close"]:
        if c not in d.columns:
            raise KeyError(f"필수 컬럼 누락: {c}")

    # TR을 pandas Series로 계산 (rolling 사용 가능하도록)
    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - d["Close"].shift(1)).abs(),
            (d["Low"]  - d["Close"].shift(1)).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(int(period)).mean()

    hl2   = (d["High"] + d["Low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    trend = np.ones(len(d), dtype=bool)  # True=상승(매수), False=하락(매도)
    for i in range(1, len(d)):
        if d["Close"].iloc[i] > upper.iloc[i - 1]:
            trend[i] = True
        elif d["Close"].iloc[i] < lower.iloc[i - 1]:
            trend[i] = False
        else:
            trend[i] = trend[i - 1]
            if trend[i]:
                lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1])
            else:
                upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1])

    out = pd.DataFrame(index=d.index)
    out["ST_up"]    = upper
    out["ST_dn"]    = lower
    out["ST_trend"] = trend  # True=상승, False=하락
    return out

# ======================================
# 백테스트 (3개 모두 매수 / 1개라도 하락 청산)
# 체결: 해당 날짜의 종가에 슬리피지 반영
# ======================================
def backtest(data: pd.DataFrame, st_params, slippage: float = 0.001, initial_capital: float = 100.0):
    st1 = supertrend(data, period=st_params[0][0], multiplier=st_params[0][1])
    st2 = supertrend(data, period=st_params[1][0], multiplier=st_params[1][1])
    st3 = supertrend(data, period=st_params[2][0], multiplier=st_params[2][1])

    buy_sig  =  st1["ST_trend"] &  st2["ST_trend"] &  st3["ST_trend"]      # 3개 모두 상승
    sell_sig = (~st1["ST_trend"]) | (~st2["ST_trend"]) | (~st3["ST_trend"]) # 1개라도 하락

    position = 0.0
    capital  = float(initial_capital)
    entry_px, entry_ts = None, None
    equity = []
    trades = []

    for i, (ts, row) in enumerate(data.iterrows()):
        px = float(row["Close"])

        # 매수(진입): 보유X & 3개 모두 상승 → 종가(+slip)
        if position == 0 and buy_sig.iloc[i]:
            entry_px = px * (1 + slippage)
            position = capital / entry_px
            capital  = 0.0
            entry_ts = ts

        # 청산: 보유O & 1개라도 하락 → 종가(-slip)
        elif position > 0 and sell_sig.iloc[i]:
            exit_px = px * (1 - slippage)
            capital = position * exit_px
            ret = (exit_px - entry_px) / entry_px
            trades.append({
                "매수일": entry_ts.strftime("%Y-%m-%d"),
                "매수가": round(entry_px, 8),
                "매도일": ts.strftime("%Y-%m-%d"),
                "매도가": round(exit_px, 8),
                "슬리피지 반영 수익률(%)": round(ret * 100, 4),
                "초기자금의 변화": round(capital, 8)
            })
            position, entry_px, entry_ts = 0.0, None, None

        # 현재 자산 평가
        equity.append(capital if position == 0 else position * px)

    # 마지막 날 보유 중이면 강제 청산(종가, -slip)
    if position > 0:
        last_px = float(data["Close"].iloc[-1]) * (1 - slippage)
        capital = position * last_px
        ret = (last_px - entry_px) / entry_px
        ts = data.index[-1]
        trades.append({
            "매수일": entry_ts.strftime("%Y-%m-%d"),
            "매수가": round(entry_px, 8),
            "매도일": ts.strftime("%Y-%m-%d"),
            "매도가": round(last_px, 8),
            "슬리피지 반영 수익률(%)": round(ret * 100, 4),
            "초기자금의 변화": round(capital, 8)
        })
        equity[-1] = capital
        position, entry_px, entry_ts = 0.0, None, None

    equity_s = pd.Series(equity, index=data.index, name="Equity")

    # 성과지표(안전 계산)
    start_v = float(equity_s.iloc[0]) if len(equity_s) > 0 else initial_capital
    end_v   = float(equity_s.iloc[-1]) if len(equity_s) > 0 else initial_capital
    days    = max((equity_s.index[-1] - equity_s.index[0]).days, 1)
    years   = days / 365.25
    total_r = end_v / start_v if start_v > 0 else np.nan
    cagr    = (total_r ** (1 / years) - 1) if pd.notna(total_r) else np.nan
    mdd     = float((equity_s / equity_s.cummax() - 1).min()) if len(equity_s) > 1 else 0.0
    rets    = equity_s.pct_change().dropna()
    sharpe  = float((rets.mean() / rets.std()) * np.sqrt(252)) if (len(rets) > 5 and rets.std() > 0) else 0.0

    trade_df = pd.DataFrame(trades)
    return equity_s, trade_df, cagr, mdd, sharpe

# ===============================
# CSV 업로드 (업비트 포맷 자동 인식)
# ===============================
uploaded = st.file_uploader("업비트 CSV 업로드 (date_utc / open / high / low / close / volume)", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)

    # 소문자 맵
    cols_lower = {c.lower(): c for c in raw.columns}

    # 필요한 컬럼 체크
    need = ["date_utc", "open", "high", "low", "close"]
    missing = [c for c in need if c not in cols_lower]
    if missing:
        st.error(f"CSV에 필요한 컬럼이 없습니다: {', '.join(missing)}")
        st.write("현재 컬럼:", list(raw.columns))
        st.stop()

    # date_utc → DatetimeIndex
    date_col = cols_lower["date_utc"]
    data = raw.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    data.index.name = "Date"

    # 숫자형 강제 변환
    for key in ["open", "high", "low", "close", "volume"]:
        if key in cols_lower:
            data[cols_lower[key]] = pd.to_numeric(data[cols_lower[key]], errors="coerce")

    # 표준 컬럼명으로 리네임
    data = data.rename(columns={
        cols_lower["open"]:  "Open",
        cols_lower["high"]:  "High",
        cols_lower["low"]:   "Low",
        cols_lower["close"]: "Close",
        **({cols_lower["volume"]: "Volume"} if "volume" in cols_lower else {})
    })

    # 필수 가격 컬럼 존재 & NaN 제거
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in data.columns else [])]

    st.success(f"✅ 로드 완료: {data.index.min().date()} ~ {data.index.max().date()} (행 {len(data):,}) — date_utc 기준, 종가 체결")

    # ===============================
    # 파라미터 입력
    # ===============================
    st.sidebar.header("Supertrend 파라미터")
    ST1_p = st.sidebar.number_input("ST1 기간", 5, 200, 10, 1)
    ST1_m = st.sidebar.number_input("ST1 배수", 0.5, 10.0, 3.0, 0.1)
    ST2_p = st.sidebar.number_input("ST2 기간", 5, 200, 20, 1)
    ST2_m = st.sidebar.number_input("ST2 배수", 0.5, 10.0, 4.0, 0.1)
    ST3_p = st.sidebar.number_input("ST3 기간", 5, 200, 30, 1)
    ST3_m = st.sidebar.number_input("ST3 배수", 0.5, 10.0, 5.0, 0.1)

    slippage_pct = st.sidebar.number_input("슬리피지(%)", 0.0, 5.0, 0.1, 0.1)
    init_cap     = st.sidebar.number_input("초기자산", 1.0, 1_000_000.0, 100.0, 1.0)  # 기본값 100
    slippage     = slippage_pct / 100.0

    # 데이터 길이 가드 (ATR 계산 여유 확보)
    max_period = max(int(ST1_p), int(ST2_p), int(ST3_p))
    if len(data) < max_period + 5:
        st.warning(f"데이터가 부족합니다. 최소 {max_period + 5}개 행 이상 필요합니다. 현재: {len(data)}")
    else:
        if st.button("🚀 백테스트 실행"):
            with st.spinner("계산 중..."):
                equity, trades, cagr, mdd, sharpe = backtest(
                    data,
                    [(int(ST1_p), float(ST1_m)), (int(ST2_p), float(ST2_m)), (int(ST3_p), float(ST3_m))],
                    slippage=slippage,
                    initial_capital=float(init_cap)
                )

            # ===== 결과 요약 =====
            st.subheader("📊 결과 요약")
            cagr_txt = "데이터 부족" if (pd.isna(cagr) or np.isinf(cagr)) else f"{cagr*100:.2f}%"
            st.write(f"**CAGR**: {cagr_txt}")
            st.write(f"**MDD** : {mdd*100:.2f}%")
            st.write(f"**Sharpe**: {sharpe:.2f}")
            st.write(f"**거래 횟수**: {len(trades)}")

            # ===== 자산 곡선 =====
            st.subheader("📈 자산 곡선 (Equity Curve)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
            fig.update_layout(template="plotly_white", xaxis_title="date_utc", yaxis_title="자산")
            st.plotly_chart(fig, use_container_width=True)

            # ===== 매매 내역 =====
            st.subheader("🧾 매매 내역 (종가 체결)")
            st.dataframe(trades)

            if not trades.empty:
                csv = trades.to_csv(index=False).encode("utf-8-sig")
                st.download_button("💾 매매 내역 다운로드", data=csv, file_name="trade_log.csv", mime="text/csv")
