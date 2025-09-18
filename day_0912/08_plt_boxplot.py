import matplotlib.pyplot as plt
import pandas as pd

# 한글 폰트 설정 및 마이너스 기호 깨짐 방지
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 사용할 한글 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

#-----------------------------------------------------------------------
# CSV 파일 읽기
df = pd.read_csv('./data/auto-mpg.csv', header=None)

# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

plt.style.use('grayscale')  # 그림 스타일 설정 (흑백 계열)

# ---------------- 수직/수평 박스플롯 (matplotlib) ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # 1행 2열 subplot

# 수직 박스플롯
ax1.boxplot(
    x=[df[df['origin']==1]['mpg'],  # USA
       df[df['origin']==2]['mpg'],  # EU
       df[df['origin']==3]['mpg']], # JAPAN
    labels=['USA', 'EU', 'JAPAN']
)

# 수평 박스플롯
ax2.boxplot(
    x=[df[df['origin']==1]['mpg'], 
       df[df['origin']==2]['mpg'],
       df[df['origin']==3]['mpg']],
    labels=['USA', 'EU', 'JAPAN'],
    vert=False  # 수평
)

# 제목 설정
ax1.set_title('제조국가별 연비 분포(수직 박스 플롯)')
ax2.set_title('제조국가별 연비 분포(수평 박스 플롯)')

# plt.show()  # 화면에 표시 (주석 처리됨)

# ---------------- 수직/수평 박스플롯 (pandas plot) ----------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 수직 박스플롯
df.plot(kind='box', column=['mpg'], by=['origin'], ax=axes[0])
axes[0].set_title('제조국가별 연비 분포(수직 박스 플롯)')
axes[0].set_xticklabels(['USA', 'EU', 'JAPAN'])  # x축 라벨 변경

# 수평 박스플롯
df.plot(kind='box', column=['mpg'], by=['origin'], ax=axes[1], vert=False)
axes[1].set_title('제조국가별 연비 분포(수평 박스 플롯)')
axes[1].set_yticklabels(['USA', 'EU', 'JAPAN'])  # y축 라벨 변경

plt.show()

# ---------------- 박스플롯 색상 지정 ----------------
color = {
    'boxes':'SeaGreen',    # 상자 색
    'whiskers':'Olive',    # 수염 색
    'medians':'red',       # 중앙값 색
    'caps':'blue'          # 상자 끝 캡 색
}

# pandas plot으로 색상 지정 수평 박스플롯
df.plot(kind='box', column=['mpg'], by=['origin'], figsize=(15, 5),
        color=color, sym='r+', vert=False)  # sym='r+' → 이상치 표시
plt.show()
