import seaborn as sns
import matplotlib.pyplot as plt

# 타이타닉 데이터셋 불러오기
titanic = sns.load_dataset('titanic')

print(titanic.head())  # 데이터셋 상위 5행 출력
print()

sns.set_style('darkgrid')  # 그래프 스타일을 'darkgrid'로 설정

# ---------------- 첫 번째 그림 ----------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 1행 2열 subplot 생성

# 선형 회귀선 포함한 산점도
sns.regplot(x='age', y='fare', data=titanic, ax=axes[0])  

# 선형 회귀선 제외한 산점도
sns.regplot(x='age', y='fare', data=titanic, ax=axes[1], fit_reg=False)  
                                         
plt.show()


# ---------------- 두 번째 그림 ----------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 1행 2열 subplot 생성

# 성별(sex)에 따라 색상을 다르게 한 산점도
sns.scatterplot(x='age', y='fare', hue='sex', data=titanic, ax=axes[0])  

# 생존 여부(survived)에 따라 색상을 다르게 한 산점도
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic, ax=axes[1])  

plt.show()
