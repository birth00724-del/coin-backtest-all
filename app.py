import pandas as pd
import pandas_ta as ta

# ✅ 예시 데이터프레임 (OHLC 데이터 필요)
# df에는 'high', 'low', 'close' 열이 있어야 합니다.
# 예: df = get_data("BTCUSDT", "1d")

# -------------------------------
# 1️⃣ 수퍼트렌드 3개 계산
# -------------------------------
st1 = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=2.0)
st2 = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
st3 = ta.supertrend(df['high'], df['low'], df['close'], length=14, multiplier=4.5)

# 데이터프레임에 병합
df = pd.concat([df, st1, st2, st3], axis=1)

# 컬럼 이름 확인 (참고용)
# print(df.columns)

# -------------------------------
# 2️⃣ 상승/하락 방향 신호 변수 설정
# -------------------------------
# pandas_ta의 SuperTrend 결과는 SUPERTd_길이_배수 형태로 생성됨 (1 = 상승, -1 = 하락)
df['ST1_dir'] = df['SUPERTd_7_2.0']
df['ST2_dir'] = df['SUPERTd_10_3.0']
df['ST3_dir'] = df['SUPERTd_14_4.5']

# -------------------------------
# 3️⃣ 세 수퍼트렌드 조건 결합
# -------------------------------
# 조건 1: 세 지표 모두 상승 → 매수 신호
df['BUY_SIGNAL'] = (
    (df['ST1_dir'] == 1) &
    (df['ST2_dir'] == 1) &
    (df['ST3_dir'] == 1)
)

# 조건 2: 세 지표 중 하나라도 하락 → 매도 신호
df['SELL_SIGNAL'] = (
    (df['ST1_dir'] == -1) |
    (df['ST2_dir'] == -1) |
    (df['ST3_dir'] == -1)
)

# -------------------------------
# 4️⃣ 포지션 상태 계산 (단순 버전)
# -------------------------------
position = 0
positions = []

for i in range(len(df)):
    if df['BUY_SIGNAL'].iloc[i]:
        position = 1  # 매수 진입
    elif df['SELL_SIGNAL'].iloc[i]:
        position = 0  # 포지션 종료
    positions.append(position)

df['POSITION'] = positions

# -------------------------------
# 5️⃣ 수익률 계산 (옵션)
# -------------------------------
df['return'] = df['close'].pct_change()
df['strategy_return'] = df['return'] * df['POSITION']

# 누적 수익률
df['cum_return'] = (1 + df['strategy_return']).cumprod()

# -------------------------------
# ✅ 결과 출력
# -------------------------------
print(df[['close', 'ST1_dir', 'ST2_dir', 'ST3_dir', 'BUY_SIGNAL', 'SELL_SIGNAL', 'POSITION', 'cum_return']].tail())
