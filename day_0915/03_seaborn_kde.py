# 커널 밀도 추정(Kernel Density Estimate, KDE) 그래프 예제
import matplotlib.pyplot as plt
import seaborn as sns

# 타이타닉 데이터셋 불러오기
titanic = sns.load_dataset('titanic')

sns.set_style('darkgrid')  # 그래프 스타일 설정

# ---------------- subplot_mosaic로 레이아웃 생성 ----------------
fig, axes = plt.subplot_mosaic(
    [['top_left', 'top_center', 'right'],
     ['bottom_left', 'bottom_center', 'right']],  # right는 두 행에 걸쳐서 차지
    figsize=(15, 6),        # 그림 크기
    constrained_layout=True  # 자동 레이아웃 조정
)

# 1) 기본 KDE 그래프
sns.kdeplot(x='age', data=titanic, ax=axes['top_left'])

# 2) 생존 여부(hue)에 따라 색상 구분
sns.kdeplot(x='age', data=titanic, hue='survived', ax=axes['bottom_left'])

# 3) fill=True → 그래프 아래 영역 색칠
sns.kdeplot(x='age', data=titanic, hue='survived', fill=True, ax=axes['top_center'])

# 4) multiple='stack' → 그룹별 밀도를 쌓아서 표시
sns.kdeplot(x='age', data=titanic, hue='survived', multiple='stack', ax=axes['bottom_center'])

# 5) multiple='fill' → 비율로 표시, bw_adjust=2.0 → 커널 폭 조절 (스무딩 정도)
sns.kdeplot(x='age', data=titanic, hue='survived', multiple='fill', bw_adjust=2.0, ax=axes['right'])

# 전체 그림 제목
fig.suptitle('Titanic - Age Distribution')

# 각 subplot 제목 설정
axes['top_left'].set_title('KDE')
axes['bottom_left'].set_title('KDE (hue)')
axes['top_center'].set_title('KDE (fill=True)')
axes['bottom_center'].set_title('KDE (multiple - stack)')
axes['right'].set_title('KDE (multiple - fill)')

plt.show()




