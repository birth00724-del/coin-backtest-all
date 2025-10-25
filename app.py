import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="SuperTrend 백테스터", layout="wide")
st.title("📈 SuperTrend 3중 조건 백테스터 (업비트 CSV 전체기간)")

# ============================================================
# 1) 데이터 로드
# ============================================================
@st.cache_data
def load_upbit_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date_kst" in df.columns:
        df["Date"] = pd.to_datetime(df["date_kst"], errors="coerce")
    elif "date_utc" in df.columns:
        df["Date"] = pd.to_datetime(df["date_utc"], errors="coerce")
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        raise KeyError("CSV에서 날짜 컬럼을 찾지 못했습니다.")

    price_col = "close" if "close" in df.columns else (
        "trade_price" if "trade_price" in df.columns else None
    )
    high_col = "high" if "high" in df.columns else (
        "candle_high_price" if "candle_high_price" in df.columns else None
    )
    low_col = "low" if "low" in df.columns else (
        "candle_low_price" if "candle_low_price" in df.columns else None
    )
    vol_col = "volume" if "volume" in df.columns else (
        "candle_acc_trade_volume" if "candle_acc_trade_volume" in df.columns else None
    )

    if None in (price_col, high_col, low_col, vol_col):
        raise KeyError("CSV에서 필요한 컬럼(close, high, low, volume)을 찾지 못했습니다.")

    df = df.rename(columns={
        price_col: "Close",
        high_col: "High",
        low_col: "Low",
        vol_col: "Volume"
    })
    df = df[["Date", "High", "Low", "Close", "Volume"]].dropna().sort_values("Date").set_index("Date")
    return df

try:
    data = load_upbit_csv("upbit_krw_btc_daily_all.csv")
    st.success(f"✅ CSV 로드 성공: {len(data):,}행 (기간: {data.index.min().date()} ~ {data.index.max().date()})")
except Exception as e:
    st.error(f"❌ CSV 로드 실패: {e}")
    st.stop()

# ============================================================
# 2) 공통 설정
# ============================================================
st.sidebar.header("⚙️ 공통 설정")
initial_capital = st.sidebar.number_input("초기자금", min_value=1.0, value=100.0, step=1.0)
slippage_pct = st.sidebar.slider("슬리피지(%)", 0.0, 5.0, 3.0, 0.5)
slippage = slippage_pct / 100.0

# ============================================================
# 3) SuperTrend 파라미터
# ============================================================
st.sidebar.header("🧪 SuperTrend 설정 (3중)")
super_params = []
for i in range(1, 4):
    with st.sidebar.expander(f"SuperTrend {i}", expanded=True):
        atr_period = st.number_input(f"ATR 기간 {i}", min_value=2, max_value=100, value=10 * i, step=1, key=f"atr_{i}")
        multiplier = st.number_input(f"Multiplier {i}", min_value=1.0, max_value=10.0, value=1.5 * i, step=0.1, key=f"mult_{i}")
        super_params.append({"atr_period": int(atr_period), "multiplier": float(multiplier)})

# ============================================================
# 4) SuperTrend 계산 함수
# ============================================================
def compute_supertrend(df, atr_period=10, multiplier=3.0):
    d = df.copy()
    hl2 = (d["High"] + d["Low"]) / 2
    d["TR"] = np.maximum.reduce([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift(1)).abs(),
        (d["Low"] - d["Close"].shift(1)).abs()
    ])
    d["ATR"] = d["TR"].rolling(atr_period).mean()
    d["upperband"] = hl2 + (multiplier * d["ATR"])
    d["lowerband"] = hl2 - (multiplier * d["ATR"])
    d["supertrend"] = np.nan
    d["trend"] = 1

    for i in range(1, len(d)):
        prev = i - 1
        if d["Close"].iloc[i] > d["upperband"].iloc[prev]:
            d["trend"].iloc[i] = 1
        elif d["Close"].iloc[i] < d["lowerband"].iloc[prev]:
            d["trend"].iloc[i] = -1
        else:
            d["trend"].iloc[i] = d["trend"].iloc[prev]
            if d["trend"].iloc[i] == 1:
                d["lowerband"].iloc[i] = max(d["lowerband"].iloc[i], d["lowerband"].iloc[prev])
            else:
                d["upperband"].iloc[i] = min(d["upperband"].iloc[i], d["upperband"].iloc[prev])

        d["supertrend"].iloc[i] = (
            d["lowerband"].iloc[i] if d["trend"].iloc[i] == 1 else d["upperband"].iloc[i]
        )
    return d["trend"]

# ============================================================
# 5) 세 SuperTrend 결합 시그널
# ============================================================
def combined_supertrend(df, params_list):
    trends = []
    for p in params_list:
        trend = compute_supertrend(df, **p)
        trends.append(trend)
    combo = np.where((trends[0] == 1) & (trends[1] == 1) & (trends[2] == 1), 1, -1)
    return pd.Series(combo, index=df.index)

# ============================================================
# 6) 백테스트
# ============================================================
signal = combined_supertrend(data, super_params)
returns = signal.shift(1) * data["Close"].pct_change()
returns -= slippage * np.abs(signal.diff().fillna(0))
curve = (1 + returns).cumprod() * initial_capital

# ============================================================
# 7) 성과 지표
# ============================================================
def metrics(curve):
    if curve.empty:
        return 0, 0, 0, 0
    final = curve.iloc[-1]
    years = max(len(curve) / 252, 1)
    cagr = (final / curve.iloc[0]) ** (1 / years) - 1
    mdd = (curve / curve.cummax() - 1).min()
    daily = curve.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() != 0 else 0
    return final, cagr, mdd, sharpe

final, cagr, mdd, sharpe = metrics(curve)

# ============================================================
# 8) 시각화
# ============================================================
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price", yaxis="y1"))
fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name="Equity Curve", yaxis="y2"))

fig.update_layout(
    title=f"SuperTrend 3중 조건 백테스트 (초기 {initial_capital:,.0f}, 슬리피지 {slippage_pct}%)",
    yaxis=dict(title="가격", side="left", showgrid=False),
    yaxis2=dict(title="자산곡선", side="right", overlaying="y", showgrid=False),
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 성과 요약")
st.write({
    "최종자산": f"{final:,.2f}",
    "CAGR": f"{cagr*100:.2f}%",
    "MDD": f"{mdd*100:.2f}%",
    "Sharpe": f"{sharpe:.2f}"
})

st.success("✅ 완료! SuperTrend 3중 조건 백테스트 실행 성공")
