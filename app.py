import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="TV-Style Supertrend + VWMA + VPVR-DVA Backtester", layout="wide")
st.title("📈 Supertrend(TradingView) 3중 결합 + VWMA 필터 + VPVR DVA (102일 롤링) — 프리셋 저장·불러오기")

# =========================================================
# 0) 유틸: 안전 클램프 / 프리셋 보정
# =========================================================
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
    if np.isnan(v) or np.isinf(v):
        v = lo
    return max(lo, min(hi, v))

FILL_OPTIONS = ["당일 종가", "다음날 시가", "다음날 종가"]

def sanitize_preset(p):
    return {
        # 전략 선택
        "use_st": bool(p.get("use_st", True)),
        "use_vwma": bool(p.get("use_vwma", False)),
        "use_vpvr": bool(p.get("use_vpvr", False)),

        # ST 파라미터
        "ST1_L": clamp_int(p.get("ST1_L", 10), 5, 200),
        "ST1_M": clamp_float(p.get("ST1_M", 3.0), 0.5, 10.0),
        "ST2_L": clamp_int(p.get("ST2_L", 20), 5, 200),
        "ST2_M": clamp_float(p.get("ST2_M", 4.0), 0.5, 10.0),
        "ST3_L": clamp_int(p.get("ST3_L", 30), 5, 200),
        "ST3_M": clamp_float(p.get("ST3_M", 5.0), 0.5, 10.0),

        # VWMA
        "VWMA_L": clamp_int(p.get("VWMA_L", 20), 2, 300),

        # VPVR
        "VPVR_BINS": clamp_int(p.get("VPVR_BINS", 64), 20, 200),  # 가로 bin 수 (102일 고정)

        # 실행 설정
        "slippage_pct": clamp_float(p.get("slippage_pct", 0.1), 0.0, 5.0),
        "init_cap": clamp_float(p.get("init_cap", 100.0), 1.0, 1_000_000.0),
        "fill_policy": p.get("fill_policy", "다음날 시가") if p.get("fill_policy", "다음날 시가") in FILL_OPTIONS else "다음날 시가",
    }

# =========================================================
# 1) Wilder RMA (TradingView ta.rma)
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
# 2) Supertrend (TradingView 방식)
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
    dir_long.iloc[0]    = True  # 시작값 임의

    for i in range(1, len(d)):
        final_upper.iloc[i] = (
            basic_upper.iloc[i] if (c.iloc[i-1] > final_upper.iloc[i-1])
            else min(basic_upper.iloc[i], final_upper.iloc[i-1])
        )
        final_lower.iloc[i] = (
            basic_lower.iloc[i] if (c.iloc[i-1] < final_lower.iloc[i-1])
            else max(basic_lower.iloc[i], final_lower.iloc[i-1])
        )

        prev_line = final_lower.iloc[i-1] if dir_long.iloc[i-1] else final_upper.iloc[i-1]
        if c.iloc[i] > prev_line:
            dir_long.iloc[i] = True
        elif c.iloc[i] < prev_line:
            dir_long.iloc[i] = False
        else:
            dir_long.iloc[i] = dir_long.iloc[i-1]

    out = pd.DataFrame(index=d.index)
    out["ST_trend"] = dir_long
    out["Upper"]    = final_upper
    out["Lower"]    = final_lower
    out["ST_line"]  = np.where(dir_long, final_lower, final_upper).astype(float)
    return out

# =========================================================
# 3) VWMA (거래량가중이동평균)
# =========================================================
def compute_vwma(df: pd.DataFrame, window: int) -> pd.Series:
    if "Volume" not in df.columns:
        return pd.Series(index=df.index, dtype=float) * np.nan
    v = pd.to_numeric(df["Volume"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    num = (c * v).rolling(window, min_periods=window).sum()
    den = v.rolling(window, min_periods=window).sum()
    return num / den

# =========================================================
# 4) VPVR DVA (102일 롤링) — DVAL/ DVAH 산출
#   - 창: 고정 102일
#   - bins: 사용자가 조절 가능 (기본 64)
#   - 방법: 창 구간에서 '종가에 모든 거래량을 귀속' (일봉 데이터 한계)
#           볼륨 프로파일로 POC 찾고, 양옆으로 확장하며 누적 70% 도달 구간 = VA
# =========================================================
def compute_vpvr_dva(df: pd.DataFrame, window: int = 102, bins: int = 64):
    if "Volume" not in df.columns:
        return pd.Series(index=df.index, dtype=float) * np.nan, pd.Series(index=df.index, dtype=float) * np.nan

    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    vol   = pd.to_numeric(df["Volume"], errors="coerce").astype(float)

    idx = df.index
    DVAL = pd.Series(index=idx, dtype=float)
    DVAH = pd.Series(index=idx, dtype=float)

    # 미리 numpy로
    c_vals = close.values
    v_vals = vol.values

    for end in range(window - 1, len(df)):
        start = end - window + 1
        c_win = c_vals[start:end+1]
        v_win = v_vals[start:end+1]
        if np.any(np.isnan(c_win)) or np.any(np.isnan(v_win)):
            continue

        lo = np.min(c_win)
        hi = np.max(c_win)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        # bin 경계
        edges = np.linspace(lo, hi, bins + 1)
        # 각 일자 종가를 해당 bin에 할당 (일봉이라 종가에 전량 귀속)
        bin_idx = np.clip(np.digitize(c_win, edges) - 1, 0, bins - 1)
        vol_hist = np.bincount(bin_idx, weights=v_win, minlength=bins).astype(float)

        total = vol_hist.sum()
        if total <= 0:
            continue

        poc = int(np.argmax(vol_hist))
        target = 0.7 * total
        cum = vol_hist[poc]
        left = poc - 1
        right = poc + 1
        min_i = poc
        max_i = poc

        while cum < target and (left >= 0 or right < bins):
            lv = vol_hist[left] if left >= 0 else -1.0
            rv = vol_hist[right] if right < bins else -1.0
            if rv > lv:
                cum += max(rv, 0.0)
                max_i = right
                right += 1
            else:
                cum += max(lv, 0.0)
                min_i = left
                left -= 1

        # VA 경계: 하한=edges[min_i], 상한=edges[max_i+1]
        DVAL.iloc[end] = float(edges[min_i])
        DVAH.iloc[end] = float(edges[max_i + 1])

    return DVAL, DVAH

# =========================================================
# 5) 백테스트 (ST + 선택적 VWMA + 선택적 VPVR-DVA)
#    - 기본(ST): 3개 모두 상승(True) → 매수 후보 / 1개라도 하락(False) → 매도 후보
#    - VWMA(선택): 매수 시 Close > VWMA 여야 함 (매도엔 미개입)
#    - VPVR-DVA(선택): 매수 시 Close > DVAH 여야 함, 또한 Close <= DVAH면 즉시 매도
#    - 체결: 선택형 (당일 종가 / 다음날 시가 / 다음날 종가)
# =========================================================
def execute_backtest(
    data: pd.DataFrame,
    st_cfgs,              # [(L1,M1),(L2,M2),(L3,M3)]
    fill_policy: str,
    slippage: float,
    initial_capital: float,
    use_vwma: bool = False,
    vwma_len: int = 20,
    use_vpvr: bool = False,
    vpvr_bins: int = 64,
):
    # Supertrend 3개
    st_frames = [supertrend_tv(data, int(L), float(M)) for (L, M) in st_cfgs]
    trends = pd.concat([f["ST_trend"] for f in st_frames], axis=1)
    trends.columns = [f"ST{i+1}" for i in range(3)]
    base_buy  = (trends.sum(axis=1) == 3)
    base_sell = (trends.sum(axis=1) < 3)

    # VWMA
    if use_vwma:
        vwma = compute_vwma(data, int(vwma_len))
        vwma_ok = data["Close"] > vwma
        buy_sig = base_buy & vwma_ok
    else:
        vwma = None
        buy_sig = base_buy

    # VPVR DVA
    if use_vpvr:
        if "Volume" not in data.columns:
            raise ValueError("VPVR DVA를 사용하려면 CSV에 'volume' 컬럼이 필요합니다.")
        dval, dvah = compute_vpvr_dva(data, window=102, bins=int(vpvr_bins))
        # 매수 필터 추가
        buy_sig = buy_sig & (data["Close"] > dvah)
        # 매도 조건에 'Close <= DVAH' 추가 (강제 청산)
        vpvr_sell = data["Close"] <= dvah
    else:
        dval = dvah = None
        vpvr_sell = pd.Series(False, index=data.index)

    # 최종 매도 시그널
    sell_sig = base_sell | vpvr_sell

    # 체결 타이밍/가격
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

    # 시뮬레이션
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

    # 마지막 강제 청산
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

    return equity_s, pd.DataFrame(trades), cagr, mdd, sharpe, st_frames, (vwma if use_vwma else None), (dval, dvah) if use_vpvr else (None, None)

# =========================================================
# 6) CSV 업로드 (업비트: date_kst/date_utc + o/h/l/c [+ volume])
# =========================================================
uploaded = st.file_uploader("업비트 CSV 업로드 (date_kst 또는 date_utc / open / high / low / close / [volume])", type=["csv"])

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

    rename_map = {
        cols_lower["open"]: "Open",
        cols_lower["high"]: "High",
        cols_lower["low"]:  "Low",
        cols_lower["close"]: "Close"
    }
    if "volume" in cols_lower:
        rename_map[cols_lower["volume"]] = "Volume"

    data = data.rename(columns=rename_map)
    keep_cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in data.columns else [])
    data = data[keep_cols].dropna(subset=["Open", "High", "Low", "Close"])

    st.success(f"✅ 로드 완료: {data.index.min().date()} ~ {data.index.max().date()} (행 {len(data):,}) — 기준: {tz_col} — 컬럼: {', '.join(keep_cols)}")

    # =====================================================
    # 7) 프리셋: 저장/불러오기(안전), 적용 타이밍 개선
    # =====================================================
    if "presets" not in st.session_state:
        st.session_state["presets"] = {}

    if "_pending_preset" in st.session_state:
        safe = sanitize_preset(st.session_state["_pending_preset"])
        # 전략 선택
        st.session_state["use_st"] = safe["use_st"]
        st.session_state["use_vwma"] = safe["use_vwma"]
        st.session_state["use_vpvr"] = safe["use_vpvr"]
        # ST
        st.session_state["ST1_L"] = safe["ST1_L"]
        st.session_state["ST1_M"] = safe["ST1_M"]
        st.session_state["ST2_L"] = safe["ST2_L"]
        st.session_state["ST2_M"] = safe["ST2_M"]
        st.session_state["ST3_L"] = safe["ST3_L"]
        st.session_state["ST3_M"] = safe["ST3_M"]
        # VWMA / VPVR
        st.session_state["VWMA_L"] = safe["VWMA_L"]
        st.session_state["VPVR_BINS"] = safe["VPVR_BINS"]
        # 실행설정
        st.session_state["slippage_pct"] = safe["slippage_pct"]
        st.session_state["init_cap"] = safe["init_cap"]
        st.session_state["fill_policy"] = safe["fill_policy"]
        del st.session_state["_pending_preset"]

    # =====================================================
    # 8) 사이드바 — 전략 선택 & 파라미터
    # =====================================================
    st.sidebar.header("🧠 전략 선택")
    use_st   = st.sidebar.checkbox("수퍼트렌드 x3 사용", value=st.session_state.get("use_st", True), key="use_st")
    use_vwma = st.sidebar.checkbox("VWMA 필터 사용 (매수: Close > VWMA)", value=st.session_state.get("use_vwma", False), key="use_vwma")
    use_vpvr = st.sidebar.checkbox("VPVR DVA 필터 사용 (창: 고정 102일)", value=st.session_state.get("use_vpvr", False), key="use_vpvr")

    st.sidebar.header("⚙️ 수퍼트렌드 파라미터")
    ST1_L = st.sidebar.number_input("ST1 기간", 5, 200, st.session_state.get("ST1_L", 10), 1, key="ST1_L")
    ST1_M = st.sidebar.number_input("ST1 배수", 0.5, 10.0, st.session_state.get("ST1_M", 3.0), 0.1, key="ST1_M")
    ST2_L = st.sidebar.number_input("ST2 기간", 5, 200, st.session_state.get("ST2_L", 20), 1, key="ST2_L")
    ST2_M = st.sidebar.number_input("ST2 배수", 0.5, 10.0, st.session_state.get("ST2_M", 4.0), 0.1, key="ST2_M")
    ST3_L = st.sidebar.number_input("ST3 기간", 5, 200, st.session_state.get("ST3_L", 30), 1, key="ST3_L")
    ST3_M = st.sidebar.number_input("ST3 배수", 0.5, 10.0, st.session_state.get("ST3_M", 5.0), 0.1, key="ST3_M")

    st.sidebar.header("⚙️ VWMA 파라미터")
    VWMA_L = st.sidebar.number_input("VWMA 기간", 2, 300, st.session_state.get("VWMA_L", 20), 1, key="VWMA_L")

    st.sidebar.header("⚙️ VPVR 파라미터")
    st.sidebar.caption("기간은 고정 102일이며, 아래는 가격 축을 나누는 bin 수입니다.")
    VPVR_BINS = st.sidebar.number_input("VPVR 가로 bin 수", 20, 200, st.session_state.get("VPVR_BINS", 64), 1, key="VPVR_BINS")

    st.sidebar.header("⚙️ 실행 설정")
    slippage_pct = st.sidebar.number_input("슬리피지(%)", 0.0, 5.0, st.session_state.get("slippage_pct", 0.1), 0.1, key="slippage_pct")
    init_cap     = st.sidebar.number_input("초기자산", 1.0, 1_000_000.0, st.session_state.get("init_cap", 100.0), 1.0, key="init_cap")
    fill_policy  = st.sidebar.radio("체결 시점", FILL_OPTIONS, index=FILL_OPTIONS.index(st.session_state.get("fill_policy", "다음날 시가")), key="fill_policy")

    slippage = st.session_state["slippage_pct"] / 100.0

    # 프리셋 저장/불러오기
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧩 프리셋 (저장/불러오기)")
    c1, c2 = st.sidebar.columns([2,1])
    preset_name = c1.text_input("프리셋 이름", placeholder="예: STx3_VWMA20_VPVR", key="preset_name")
    save_btn    = c2.button("저장", use_container_width=True)

    def current_params():
        return {
            "use_st": bool(st.session_state["use_st"]),
            "use_vwma": bool(st.session_state["use_vwma"]),
            "use_vpvr": bool(st.session_state["use_vpvr"]),
            "ST1_L": int(st.session_state["ST1_L"]),
            "ST1_M": float(st.session_state["ST1_M"]),
            "ST2_L": int(st.session_state["ST2_L"]),
            "ST2_M": float(st.session_state["ST2_M"]),
            "ST3_L": int(st.session_state["ST3_L"]),
            "ST3_M": float(st.session_state["ST3_M"]),
            "VWMA_L": int(st.session_state["VWMA_L"]),
            "VPVR_BINS": int(st.session_state["VPVR_BINS"]),
            "slippage_pct": float(st.session_state["slippage_pct"]),
            "init_cap": float(st.session_state["init_cap"]),
            "fill_policy": st.session_state["fill_policy"],
        }

    if save_btn and preset_name.strip():
        st.session_state["presets"][preset_name.strip()] = current_params()
        st.sidebar.success(f"저장됨: {preset_name.strip()}")

    opt_keys = ["(선택)"] + list(st.session_state["presets"].keys())
    sel = st.sidebar.selectbox("프리셋 불러오기", options=opt_keys, index=0, key="preset_select")
    apply_btn = st.sidebar.button("불러오기/적용", use_container_width=True)

    if apply_btn and sel != "(선택)":
        p = st.session_state["presets"][sel]
        st.session_state["_pending_preset"] = sanitize_preset(p)
        st.sidebar.success(f"적용 준비됨: {sel}")
        st.rerun()

    # ================= 실행 =================
    if not use_st and (use_vwma or use_vpvr):
        st.warning("VWMA/VPVR는 **필터**이므로, 수퍼트렌드 x3와 함께 사용하세요.")
    else:
        # 데이터 길이 가드 (VPVR은 102일 필요)
        need_len = max(int(ST1_L), int(ST2_L), int(ST3_L), 102 if use_vpvr else 2, int(VWMA_L) if use_vwma else 2)
        if len(data) < need_len + 10:
            st.warning(f"데이터가 부족합니다. 최소 {need_len + 10}개 행 이상 필요합니다.")
        else:
            if st.button("🚀 백테스트 실행"):
                if (use_vwma or use_vpvr) and "Volume" not in data.columns:
                    st.error("VWMA/VPVR을 사용하려면 CSV에 'volume' 컬럼이 필요합니다.")
                else:
                    with st.spinner("계산 중..."):
                        equity, trades, cagr, mdd, sharpe, st_frames, vwma_s, (dval_s, dvah_s) = execute_backtest(
                            data,
                            [(ST1_L, ST1_M), (ST2_L, ST2_M), (ST3_L, ST3_M)],
                            fill_policy=st.session_state["fill_policy"],
                            slippage=slippage,
                            initial_capital=float(st.session_state["init_cap"]),
                            use_vwma=use_vwma,
                            vwma_len=int(VWMA_L),
                            use_vpvr=use_vpvr,
                            vpvr_bins=int(VPVR_BINS),
                        )

                    # 결과 요약
                    st.subheader("📊 결과 요약")
                    cagr_txt = "데이터 부족" if (pd.isna(cagr) or np.isinf(cagr)) else f"{cagr*100:.2f}%"
                    st.write(f"**CAGR**: {cagr_txt}")
                    st.write(f"**MDD** : {mdd*100:.2f}%")
                    st.write(f"**Sharpe**: {sharpe:.2f}")
                    st.write(f"**거래 횟수**: {len(trades)}")

                    # 가격 + ST 라인 (+ VWMA + VPVR DVAL/DVAH)
                    st.subheader("📈 가격 & Supertrend (TV) " + ("+ VWMA" if use_vwma else "") + (" + VPVR DVA(102일)" if use_vpvr else ""))
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
                    if use_vwma and vwma_s is not None:
                        fig.add_trace(go.Scatter(x=vwma_s.index, y=vwma_s.values, mode="lines", name=f"VWMA({int(VWMA_L)})", line=dict(width=2, color="#1565c0")))
                    if use_vpvr and (dval_s is not None) and (dvah_s is not None):
                        fig.add_trace(go.Scatter(x=dval_s.index, y=dval_s.values, mode="lines", name="DVAL(102d)", line=dict(width=1, color="#455a64", dash="dash")))
                        fig.add_trace(go.Scatter(x=dvah_s.index, y=dvah_s.values, mode="lines", name="DVAH(102d)", line=dict(width=2, color="#1e88e5")))

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
