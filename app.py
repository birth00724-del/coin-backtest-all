import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="TV-Style Supertrend Backtester (Preset Save/Load)", layout="wide")
st.title("📈 Supertrend (TradingView 호환) — 3중 결합 / KST 기준 / 프리셋 저장·불러오기")

# =========================================================
# 0) Wilder RMA (TradingView ta.rma)
# =========================================================
def rma(series: pd.Series, length: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    r = pd.Series(index=s.index, dtype=float)
    if len(s) < length:
        return s * np.nan
    r.iloc[length - 1] = s.iloc[:length].mean()
    alpha = 1.0 / float(length)
    for i in range(length, len(s)):
        r.iloc[i] = r.iloc[i - 1] + alpha * (s.iloc[i] - r.iloc[i - 1])
    return r

# =========================================================
# 1) Supertrend (TradingView 방식)
# =========================================================
def supertrend_tv(df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
    d = df.copy()
    h, l, c = d["High"], d["Low"], d["Close"]

    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = rma(tr, int(length))
    hl2 = (h + l) / 2.0

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(index=d.index, dtype=float)
    final_lower = pd.Series(index=d.index, dtype=float)
    dir_long    = pd.Series(index=d.index, dtype=bool)

    final_upper.iloc[0] = basic_upper.iloc[0]
    final_lower.iloc[0] = basic_lower.iloc[0]
    dir_long.iloc[0]    = True

    for i in range(1, len(d)):
        final_upper.iloc[i] = basic_upper.iloc[i] if (c.iloc[i-1] > final_upper.iloc[i-1]) else min(basic_upper.iloc[i], final_upper.iloc[i-1])
        final_lower.iloc[i] = basic_lower.iloc[i] if (c.iloc[i-1] < final_lower.iloc[i-1]) else max(basic_lower.iloc[i], final_lower.iloc[i-1])

        prev_line = final_lower.iloc[i-1] if dir_long.iloc[i-1] else final_upper.iloc[i-1]
        if c.iloc[i] > prev_line:
            dir_long.iloc[i] = True
        elif c.iloc[i] < prev_line:
            dir_long.iloc[i] = False
        else:
            dir_long.iloc[i] = dir_long.iloc[i-1]

    out = pd.DataFrame(index=d.index)
    out["ST_trend"] = dir_long         # True=상승, False=하락
    out["Upper"]    = final_upper
    out["Lower"]    = final_lower
    out["ST_line"]  = np.where(dir_long, final_lower, final_upper).astype(float)
    return out

# =========================================================
# 2) 백테스트 (조건 고정)
#    - 매수: 3개 모두 상승(True)
#    - 매도: 1개라도 하락(False)
#    - 신호는 당일 종가에서 확정
#    - 체결: 옵션 (당일 종가 / 다음날 시가 / 다음날 종가)
# =========================================================
def execute_backtest(data, st_cfgs, fill_policy: str, slippage: float, initial_capital: float):
    st_frames = [supertrend_tv(data, int(L), float(M)) for (L, M) in st_cfgs]
    trends = pd.concat([f["ST_trend"] for f in st_frames], axis=1)
    trends.columns = [f"ST{i+1}" for i in range(3)]

    # 고정된 조건
    buy_sig  = (trends.sum(axis=1) == 3)      # 3개 모두 True
    sell_sig = (trends.sum(axis=1) < 3)       # 1개라도 False

    if fill_policy == "당일 종가":
        buy_exec  = buy_sig.copy()
        sell_exec = sell_sig.copy()
        buy_px_s  = data["Close"] * (1 + slippage)
        sell_px_s = data["Close"] * (1 - slippage)
    elif fill_policy == "다음날 시가":
        buy_exec  = buy_sig.shift(1)
        sell_exec = sell_sig.shift(1)
        buy_px_s  = data["Open"] * (1 + slippage)
        sell_px_s = data["Open"] * (1 - slippage)
    else:  # 다음날 종가
        buy_exec  = buy_sig.shift(1)
        sell_exec = sell_sig.shift(1)
        buy_px_s  = data["Close"] * (1 + slippage)
        sell_px_s = data["Close"] * (1 - slippage)

    position = 0.0
    capital  = float(initial_capital)
    entry_px, entry_ts = None, None
    equity = []
    trades = []

    for ts, row in data.iterrows():
        px_c = float(row["Close"])
        bpx  = float(buy_px_s.loc[ts])  if not pd.isna(buy_px_s.loc[ts])  else np.nan
        spx  = float(sell_px_s.loc[ts]) if not pd.isna(sell_px_s.loc[ts]) else np.nan

        # 진입
        if position == 0 and buy_exec.loc[ts] == True and not np.isnan(bpx):
            entry_px = bpx
            position = capital / entry_px
            capital  = 0.0
            entry_ts = ts

        # 청산
        elif position > 0 and sell_exec.loc[ts] == True and not np.isnan(spx):
            exit_px  = spx
            capital  = position * exit_px
            ret      = (exit_px - entry_px) / entry_px
            trades.append({
                "매수일": entry_ts.strftime("%Y-%m-%d"),
                "매수가": round(entry_px, 6),
                "매도일": ts.strftime("%Y-%m-%d"),
                "매도가": round(exit_px, 6),
                "슬리피지 반영 수익률(%)": round(ret * 100, 4),
                "초기자금의 변화": round(capital, 6)
            })
            position, entry_px, entry_ts = 0.0, None, None

        equity.append(capital if position == 0 else position * px_c)

    # 마지막 강제 청산(보수적)
    if position > 0:
        last_px = float(data["Close"].iloc[-1]) * (1 - slippage)
        capital = position * last_px
        ret     = (last_px - entry_px) / entry_px
        ts      = data.index[-1]
        trades.append({
            "매수일": entry_ts.strftime("%Y-%m-%d"),
            "매수가": round(entry_px, 6),
            "매도일": ts.strftime("%Y-%m-%d"),
            "매도가": round(last_px, 6),
            "슬리피지 반영 수익률(%)": round(ret * 100, 4),
            "초기자금의 변화": round(capital, 6)
        })
        equity[-1] = capital

    equity_s = pd.Series(equity, index=data.index, name="Equity")

    # 성과
    if len(equity_s) >= 2:
        start_v, end_v = float(equity_s.iloc[0]), float(equity_s.iloc[-1])
        days  = max((equity_s.index[-1] - equity_s.index[0]).days, 1)
        years = days / 365.25
        total_r = end_v / start_v if start_v > 0 else np.nan
        cagr    = (total_r ** (1 / years) - 1) if pd.notna(total_r) else np.nan
        mdd     = float((equity_s / equity_s.cummax() - 1).min())
        rets    = equity_s.pct_change().dropna()
        sharpe  = float((rets.mean() / rets.std()) * np.sqrt(252)) if (len(rets) > 5 and rets.std() > 0) else 0.0
    else:
        cagr = mdd = sharpe = np.nan

    return equity_s, pd.DataFrame(trades), cagr, mdd, sharpe, st_frames

# =========================================================
# 3) CSV 업로드 (업비트: date_kst/date_utc + o/h/l/c)
#    └ 차트와 맞추려면 KST(date_kst) 추천
# =========================================================
uploaded = st.file_uploader("업비트 CSV 업로드 (date_kst 또는 date_utc / open / high / low / close)", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    cols_lower = {c.lower(): c for c in raw.columns}

    tz_col = "date_kst" if "date_kst" in cols_lower else ("date_utc" if "date_utc" in cols_lower else None)
    if tz_col is None:
        st.error("CSV에 date_kst 혹은 date_utc 컬럼이 필요합니다.")
        st.stop()

    # 숫자형 변환
    for key in ["open", "high", "low", "close", "volume"]:
        if key in cols_lower:
            raw[cols_lower[key]] = pd.to_numeric(raw[cols_lower[key]], errors="coerce")

    # 인덱스
    dt = pd.to_datetime(raw[cols_lower[tz_col]], errors="coerce")
    data = raw.loc[dt.notna()].copy()
    data.index = pd.to_datetime(data[cols_lower[tz_col]])
    data.index.name = "Date"
    data = data.sort_index()

    # 표준 컬럼
    need_price = ["open", "high", "low", "close"]
    missing = [k for k in need_price if k not in cols_lower]
    if missing:
        st.error(f"CSV에 필요한 컬럼이 없습니다: {', '.join(missing)}")
        st.stop()

    data = data.rename(columns={
        cols_lower["open"]: "Open",
        cols_lower["high"]: "High",
        cols_lower["low"]:  "Low",
        cols_lower["close"]: "Close"
    })
    data = data[["Open", "High", "Low", "Close"]].dropna()

    st.success(f"✅ 로드 완료: {data.index.min().date()} ~ {data.index.max().date()} (행 {len(data):,}) — 기준: {tz_col}")

    # =====================================================
    # 4) 사이드바 (조건 고정) + 프리셋: 저장/불러오기 전용
    # =====================================================
    st.sidebar.header("⚙️ 지표/실행 설정 (조건은 고정)")
    # 위젯(조건은 고정이라 표시만): 3개 모두 매수 진입 / 1개라도 매도면 청산
    st.sidebar.caption("매수·매도 조건은 고정: 3개 모두 매수 → 진입, 1개라도 매도 → 청산")

    # 위젯 키 지정 (프리셋 주입용)
    ST1_L = st.sidebar.number_input("ST1 기간", 5, 200, 10, 1, key="ST1_L")
    ST1_M = st.sidebar.number_input("ST1 배수", 0.5, 10.0, 3.0, 0.1, key="ST1_M")
    ST2_L = st.sidebar.number_input("ST2 기간", 5, 200, 20, 1, key="ST2_L")
    ST2_M = st.sidebar.number_input("ST2 배수", 0.5, 10.0, 4.0, 0.1, key="ST2_M")
    ST3_L = st.sidebar.number_input("ST3 기간", 5, 200, 30, 1, key="ST3_L")
    ST3_M = st.sidebar.number_input("ST3 배수", 0.5, 10.0, 5.0, 0.1, key="ST3_M")

    slippage_pct = st.sidebar.number_input("슬리피지(%)", 0.0, 5.0, 0.1, 0.1, key="slippage_pct")
    init_cap     = st.sidebar.number_input("초기자산", 1.0, 1_000_000.0, 100.0, 1.0, key="init_cap")
    fill_policy  = st.sidebar.radio("체결 시점", ["당일 종가", "다음날 시가", "다음날 종가"], index=1, key="fill_policy")

    slippage = st.session_state["slippage_pct"] / 100.0

    # ====== (교체) 프리셋 불러오기/적용 버튼 처리 ======
if apply_btn and sel != "(선택)":
    p = st.session_state["presets"][sel]

    # 위젯 제약 정의
    FILL_OPTIONS = ["당일 종가", "다음날 시가", "다음날 종가"]

    def clamp_int(v, lo, hi):
        try:
            v = int(round(float(v)))
        except Exception:
            v = lo
        return max(lo, min(hi, v))

    def clamp_float(v, lo, hi):
        try:
            v = float(v)
        except Exception:
            v = lo
        # 경계 내로
        if np.isnan(v) or np.isinf(v):
            v = lo
        return max(lo, min(hi, v))

    # 1) 기간(정수, [5,200])
    st.session_state["ST1_L"] = clamp_int(p.get("ST1_L", 10), 5, 200)
    st.session_state["ST2_L"] = clamp_int(p.get("ST2_L", 20), 5, 200)
    st.session_state["ST3_L"] = clamp_int(p.get("ST3_L", 30), 5, 200)

    # 2) 배수(실수, [0.5,10.0])
    st.session_state["ST1_M"] = clamp_float(p.get("ST1_M", 3.0), 0.5, 10.0)
    st.session_state["ST2_M"] = clamp_float(p.get("ST2_M", 4.0), 0.5, 10.0)
    st.session_state["ST3_M"] = clamp_float(p.get("ST3_M", 5.0), 0.5, 10.0)

    # 3) 슬리피지%(실수, [0,5])
    st.session_state["slippage_pct"] = clamp_float(p.get("slippage_pct", 0.1), 0.0, 5.0)

    # 4) 초기자산(실수, [1, 1_000_000])
    st.session_state["init_cap"] = clamp_float(p.get("init_cap", 100.0), 1.0, 1_000_000.0)

    # 5) 체결시점(라디오 옵션 강제)
    fp = p.get("fill_policy", "다음날 시가")
    if fp not in FILL_OPTIONS:
        fp = "다음날 시가"
    st.session_state["fill_policy"] = fp

    st.sidebar.success(f"적용됨: {sel}")
    st.rerun()


    # ================= 실행 =================
    max_len = max(int(ST1_L), int(ST2_L), int(ST3_L))
    if len(data) < max_len + 10:
        st.warning(f"데이터가 부족합니다. 최소 {max_len + 10}개 행 이상 필요합니다.")
    else:
        if st.button("🚀 백테스트 실행"):
            with st.spinner("계산 중..."):
                equity, trades, cagr, mdd, sharpe, st_frames = execute_backtest(
                    data,
                    [(ST1_L, ST1_M), (ST2_L, ST2_M), (ST3_L, ST3_M)],
                    fill_policy=st.session_state["fill_policy"],
                    slippage=slippage,
                    initial_capital=float(st.session_state["init_cap"])
                )

            # 결과 요약
            st.subheader("📊 결과 요약")
            cagr_txt = "데이터 부족" if (pd.isna(cagr) or np.isinf(cagr)) else f"{cagr*100:.2f}%"
            st.write(f"**CAGR**: {cagr_txt}")
            st.write(f"**MDD** : {mdd*100:.2f}%")
            st.write(f"**Sharpe**: {sharpe:.2f}")
            st.write(f"**거래 횟수**: {len(trades)}")

            # 가격 + ST 라인
            st.subheader("📈 가격 & Supertrend (TV 방식)")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False
            ))
            colors = ["#2e7d32", "#8e24aa", "#ef6c00"]
            for i, stf in enumerate(st_frames):
                fig.add_trace(go.Scatter(x=data.index, y=stf["Upper"], mode="lines", name=f"ST{i+1} Upper", line=dict(width=1, dash="dot", color=colors[i])))
                fig.add_trace(go.Scatter(x=data.index, y=stf["Lower"], mode="lines", name=f"ST{i+1} Lower", line=dict(width=1, dash="dot", color=colors[i])))
                fig.add_trace(go.Scatter(x=data.index, y=stf["ST_line"], mode="lines", name=f"ST{i+1} Line",  line=dict(width=2, color=colors[i])))
            fig.update_layout(template="plotly_white", xaxis_title=("date_kst" if "date_kst" in cols_lower else "date_utc"), yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # 자산 곡선
            st.subheader("💰 자산 곡선 (Equity)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=equity.index, y=equity.values, mode='lines', name='Equity'))
            fig2.update_layout(template="plotly_white", xaxis_title=("date_kst" if "date_kst" in cols_lower else "date_utc"), yaxis_title="Equity")
            st.plotly_chart(fig2, use_container_width=True)

            # 매매 내역
            st.subheader("🧾 매매 내역")
            st.dataframe(trades)
            if not trades.empty:
                csv = trades.to_csv(index=False).encode("utf-8-sig")
                st.download_button("💾 매매 내역 다운로드", data=csv, file_name="trade_log.csv", mime="text/csv")
