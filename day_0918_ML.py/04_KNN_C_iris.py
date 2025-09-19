# 라이브러리 불러오기
import seaborn as sns                   # seaborn: 데이터셋 로드 및 시각화
from sklearn.model_selection import train_test_split  # 데이터 분할용
from sklearn.neighbors import KNeighborsClassifier    # KNN 알고리즘
import matplotlib.pyplot as plt         # 그래프 시각화

# iris(붓꽃) 데이터셋 불러오기
df = sns.load_dataset('iris')           # 꽃받침/꽃잎 길이, 너비 + 품종(species)
print(df)                               # 데이터 전체 출력
print()

df.info()                               # 데이터 구조 및 컬럼 정보 확인

'''
붓꽃 데이터 설명
- 1936년 피셔(Fisher) 논문에서 처음 사용된 유명한 데이터셋
- 총 150개 샘플 (품종별 50개씩)
특징(Feature):
    sepal_length : 꽃받침 길이(cm)
    sepal_width  : 꽃받침 너비(cm)
    petal_length : 꽃잎 길이(cm)
    petal_width  : 꽃잎 너비(cm)
타깃(Target):
    species(품종) : Setosa / Versicolor / Virginica
'''

# 데이터 분포 시각화: 꽃잎 길이 vs 꽃잎 너비
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
plt.show()

# 입력값(x), 타깃값(y) 준비
X = df.drop("species", axis=1)   # 입력 데이터 (꽃받침/꽃잎 길이, 너비)
y = df["species"]                # 타깃 데이터 (품종)

print("---------------------")
print(X)                         # 입력 데이터 출력
print("---------------------")
print(y)                         # 타깃 데이터 출력

# 학습용/테스트용 데이터 분할
train_input, test_input, train_target, test_target = train_test_split(
    X, y, test_size=0.2, random_state=42  
    # test_size=0.2 → 전체 데이터의 20%를 테스트 데이터로 사용
    # random_state=42 → 시드값 고정(결과 재현 가능)
)

# KNN 분류기 생성 (이웃 개수 = 3)
kn = KNeighborsClassifier(n_neighbors=3)

# 모델 학습
kn.fit(train_input, train_target)   # 훈련 데이터로 학습
print('kn 모델 변수에 학습 완료')
print()

# 학습된 모델로 테스트 데이터 예측
y_pred = kn.predict(test_input)

# 예측 결과 vs 실제값 출력
print("예측값:", y_pred)            # 모델이 예측한 품종
print("실제값:", test_target)       # 실제 품종(정답)
print('스코어:', kn.score(test_input, test_target))

from sklearn.metrics import accuracy_score

print('정확도:', accuracy_score(test_target,y_pred))

