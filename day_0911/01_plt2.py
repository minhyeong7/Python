import matplotlib.pyplot as plt
import pandas as pd
import numpy as np   # 배열 생성용

from matplotlib import font_manager, rc
# ---------------- 한글 폰트 설정 ----------------
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# ---------------- 음수 표시 ----------------
plt.rcParams['axes.unicode_minus'] = False       # 음수 기호(-) 깨짐 방지

# -----------------------------------------------------------------------
# 1. 기본 단일 그래프

fig, ax = plt.subplots(1,1,figsize=(8,6))      # 1행 1열 subplot, 크기 8x6
ax.plot(
    range(1,10),                                # x값 1~9
    range(11,29,2),                             # y값 11,13,15,...27
    marker='x',                                 # 데이터 포인트 마커 'x'
    label='내그래프'                             # 범례 라벨
)
ax.set_title('굉장한 그래프')                   # 그래프 제목
ax.set_xlabel('길다란 가로축')                 # x축 레이블
ax.set_ylabel('놀라운 y축')                     # y축 레이블
plt.legend()                                   # 범례 표시
# plt.show()

# -----------------------------------------------------------------------
# 2. 2x2 서브플롯 예제
fig, axes = plt.subplots(2,2, figsize=(10,8))   # 2행 2열 subplot, 크기 10x8

# --- (0,0) 첫 번째 subplot ---
axes[0,0].plot(
    range(1,10), range(11,20), marker='s', label='일번연결'
)                                               # 사각형 마커
axes[0,0].set_title('일번그래프')               # 제목
axes[0,0].set_ylabel('y')                       # y축 레이블
axes[0,0].set_xlabel('x')                       # x축 레이블
axes[0,0].legend()                              # 범례

# --- (0,1) 두 번째 subplot ---
a = [7,8,9] 
sr1 = pd.Series(a)                               # pandas Series 생성
axes[0,1].plot(sr1)                              # Series plot → index가 x, 값이 y

# --- (1,0) 세 번째 subplot ---
sr1.plot(ax=axes[1,0], color='orange')           # Series를 subplot에 그림, 색상 주황
axes[1,0].set_xticks(range(0,11,1))             # x축 눈금: 0~10, 간격 1
axes[1,0].set_yticks(range(0, 101, 10))         # y축 눈금: 0~100, 간격 10

# --- (1,1) 네 번째 subplot ---
c = range(10,101,10)                            # 10,20,...100
d = np.arange(100,9,-10)                        # 100,90,...10
df = pd.DataFrame({'숫자1':c, '숫자2':d}, index=range(10,101,10))  
# DataFrame 생성, index = 10,20,...100
print(df)

axes[1,1].plot(df)                              # DataFrame plot → 각 컬럼별 선그래프

# -----------------------------------------------------------------------
plt.tight_layout()                               # subplot 간격 자동 조정 (겹치지 않게)
plt.show()                                       # 그래프 출력
