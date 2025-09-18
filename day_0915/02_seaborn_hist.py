import matplotlib.pyplot as plt
import seaborn as sns

# 타이타닉 데이터셋 불러오기
titanic = sns.load_dataset('titanic')

sns.set_style('darkgrid')  # 그래프 스타일 설정

# ---------------- subplot_mosaic 사용하여 레이아웃 생성 ----------------
fig, axes = plt.subplot_mosaic(
    [['top_left', 'top_right'],
     ['middle_left', 'middle_right'],
     ['bottom', 'bottom']],  # 레이아웃 지정, bottom은 한 행 전체 차지
    figsize=(15, 6),           # 그림 크기 지정
    constrained_layout=True     # 자동 레이아웃 조정
)

# 1) 기본 히스토그램
sns.histplot(x='age', data=titanic, bins=10, ax=axes['top_left'])

# 2) 생존 여부에 따라 색상 구분
sns.histplot(x='age', hue='survived', data=titanic, ax=axes['top_right'])

# 3) 여러 그룹을 옆으로 나란히(dodge) 표시
sns.histplot(x='age', hue='survived', multiple='dodge', 
             data=titanic, palette='muted', ax=axes['middle_left'])
# palette: 색상 테마, 예: set1, set2, pastel, muted, deep, dark, bright

# 4) 여러 그룹을 쌓아서(stack) 표시
sns.histplot(x='age', hue='survived', multiple='stack', data=titanic, ax=axes['middle_right'])

# 5) 비율로(normalized) 표시 (fill)
sns.histplot(x='age', hue='survived', multiple='fill', bins=10, data=titanic, ax=axes['bottom'])

# 전체 그림 제목
fig.suptitle('Titanic - Age Distribution')

# 각 subplot 제목 설정
axes['top_left'].set_title('Histogram')
axes['top_right'].set_title('Histogram (hue)')
axes['middle_left'].set_title('Histogram (multiple - dodge)')
axes['middle_right'].set_title('Histogram (multiple - stack)')
axes['bottom'].set_title('Histogram (multiple - fill)')

plt.show()
