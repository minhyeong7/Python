import pandas as pd

# 농어의 길이, 높이, 두께 데이터 불러오기
perch_full = pd.read_csv('./data/perch_data.csv')
print(perch_full.head())  # 앞부분 5줄 확인

perch_full.info()  # 컬럼 정보 및 데이터 타입 확인

# 인풋 데이터 - length height width

import numpy as np

# 타깃 데이터 (무게)
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
)

from sklearn.model_selection import train_test_split

# 훈련/테스트 데이터 분할
train_input, test_input, train_target, test_target= \
train_test_split(perch_full, perch_weight, random_state=42)

from sklearn.preprocessing import PolynomialFeatures

# ------------------- 실제 데이터에 적용 -------------------

# 디폴트 degree=2 (2제곱, 교차항 포함)
poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)  # 원본 데이터(길이, 높이, 두께)에 대해 학습
train_poly = poly.transform(train_input)  # 변환 → 제곱, 교차항 추가됨

print('인풋 데이터 2제곱 특성공학')
print(train_poly)  # 실제 변환된 데이터 출력
print(train_poly.shape)  # (훈련데이터 개수, 특성 개수)
print(poly.get_feature_names_out())  
# 새로운 특성 이름 확인 (길이^2, 높이^2, 길이*높이 ...)

# 테스트 데이터도 특성공학 적용
test_poly = poly.transform(test_input)  # 테스트 데이터도 같은 변환 적용

# --------- 스케일링 -----------------
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# ----------- 릿지 모델 --------------
# 손실함수 = MSE + L2 정규항

from sklearn.linear_model import Ridge   # 리지 회귀 모델 불러오기

# 기본 리지 회귀 모델(α=1.0 기본값) 생성
ridge = Ridge()

# 훈련 데이터로 학습
ridge.fit(train_scaled, train_target)

print('릿지회귀 훈련/테스트 스코어')
# 훈련 데이터에 대한 결정계수(R^2) 출력
print(ridge.score(train_scaled, train_target))
# 테스트 데이터에 대한 결정계수(R^2) 출력
print(ridge.score(test_scaled, test_target))


# ==============================================
# 최적의 규제값(alpha) 찾기
# ==============================================
import matplotlib.pyplot as plt

train_score = []   # 훈련 세트 성능 저장용 리스트
test_score = []    # 테스트 세트 성능 저장용 리스트

# 실험할 alpha 값 리스트 (규제 강도)
alpha_list = [0.0001, 0.01, 0.1, 1, 10, 100]

# alpha 값을 하나씩 바꿔가며 학습 및 평가
for alpha in alpha_list:
    # 각 alpha 값을 적용한 Ridge 모델 생성
    ridge = Ridge(alpha=alpha)
    # 모델 학습 (규제 적용됨)
    ridge.fit(train_scaled, train_target)
    # 훈련 데이터 점수 저장
    train_score.append(ridge.score(train_scaled, train_target))
    # 테스트 데이터 점수 저장
    test_score.append(ridge.score(test_scaled, test_target))

# ==============================================
# 시각화
# ==============================================
plt.plot(alpha_list, train_score, label='train')   # 훈련 점수 그래프
plt.plot(alpha_list, test_score, label='test')     # 테스트 점수 그래프
plt.xscale('log')    # x축을 로그 스케일로 (0.0001 → 100 범위 효과적으로 보기 위함)
plt.xlabel('alpha')  # x축 이름
plt.ylabel('R^2')    # y축 이름 (성능 지표)
plt.legend()         # 범례 표시
plt.show()           # 그래프 출력


ridge= Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)

print('릿지회귀 규제 0.1 훈련/테스트 스코어')
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))






