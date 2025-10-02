# ---------- 와인 데이터 로드 및 확인 ----------
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')   # 와인 데이터셋 불러오기
print(wine.head())                                  # 앞부분 데이터 확인
wine.info()                                         # 데이터 정보 확인 (결측치, 타입 등)
print()
print(wine.describe())                              # 수치형 컬럼 통계값 확인
print()

# 입력 데이터 (특성)과 타겟(라벨) 분리
data = wine[['alcohol', 'sugar', 'pH']]  # 특성 3개 선택
target = wine['class']                   # 타겟 = 와인 종류 (0: 레드, 1: 화이트)
print(target.unique())                   # 고유한 클래스 확인
print()

# ---------- 훈련/테스트 셋 분리 ----------
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)   # 8:2 비율로 나누기

print(train_input.shape, test_input.shape)          # 훈련/테스트 데이터 크기 확인
print()

# ---------- 표준화 (스케일링) ----------
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)                                 # 훈련 데이터 기준으로 스케일링 학습

train_scaled = ss.transform(train_input)            # 훈련 데이터 변환
test_scaled = ss.transform(test_input)              # 테스트 데이터 변환

# ---------- 로지스틱 회귀 ----------
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)                  # 모델 훈련

print('로지스틱 리그레션 훈련/테스트 스코어')
print(lr.score(train_scaled, train_target))         # 훈련 정확도
print(lr.score(test_scaled, test_target))           # 테스트 정확도
print('파라미터 결과')
print(lr.coef_, lr.intercept_)                      # 학습된 가중치와 절편
print()

# ---------- 결정 트리 ----------
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)        # 기본 결정트리 모델
dt.fit(train_scaled, train_target)                  # (스케일링 된 데이터로 학습)

print('결정트리 훈련/테스트 스코어')
print(dt.score(train_scaled, train_target))         # 훈련 정확도
print(dt.score(test_scaled, test_target))           # 테스트 정확도

# ---------- 결정 트리 시각화 ----------
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)                                       # 기본 트리 출력
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True,             # 깊이 1까지만 시각화
          feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# ---------- 트리 깊이를 3으로 제한 ----------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)  # 가지치기 (과적합 방지)
dt.fit(train_scaled, train_target)

print('깊이 3 나무 스코어')
print(dt.score(train_scaled, train_target))         # 훈련 정확도
print(dt.score(test_scaled, test_target))           # 테스트 정확도

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# ---------- 사실은 결정트리는 스케일링이 필요 없음 ----------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)                   # 원본 데이터 그대로 학습

print('노 스케일링 스코어')
print(dt.score(train_input, train_target))          # 훈련 정확도
print(dt.score(test_input, test_target))            # 테스트 정확도

# ---------- 특성 중요도 확인 ----------
print('----- 특성별 중요도 -----')
print(dt.feature_importances_)                      # 각 특성(alcohol, sugar, pH)의 중요도

# ---------- 정보이득 기준(min_impurity_decrease) 적용 ----------
dt = DecisionTreeClassifier(min_impurity_decrease=0.0005, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))          # 훈련 정확도
print(dt.score(test_input, test_target))            # 테스트 정확도

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


# ---------- 물고기 데이터 ----------
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())
print()

# 물고기 종류 확인 (7종류 있음)
print(pd.unique(fish['Species']))
print()

# 입력 데이터 (특성들)
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']]
print(fish_input.head())
print()

# 타겟 데이터 (종류)
fish_target = fish['Species']

# 훈련/테스트 데이터 분리 (기본 비율 75:25)
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# ---------- 다중 분류 결정트리 ----------
dt = DecisionTreeClassifier(max_depth=50, random_state=42)  # 깊이 제한 크게 설정
dt.fit(train_input, train_target)

print('다중분류 스코어')
print(dt.score(train_input, train_target))          # 훈련 정확도
print(dt.score(test_input, test_target))            # 테스트 정확도

# 트리 시각화 (너무 복잡하므로 주석 처리됨)
# plt.figure(figsize=(20,15))
# plot_tree(dt, filled=True, feature_names=['Weight', 'Length', 'Diagonal', 'Height', 'Width'])
# plt.show()
