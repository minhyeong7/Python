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

# ------------------- PolynomialFeatures 개념 확인 -------------------

# 제곱, 교차항, 상수항(1)까지 포함
poly = PolynomialFeatures()
poly.fit([[2,3]])
print(poly.transform([[2,3]]))  
# [1, 2, 3, 4, 6, 9]  → 1(상수항), x1, x2, x1^2, x1*x2, x2^2

# include_bias=False → 상수항(1) 제거
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))  
# [2, 3, 4, 6, 9]  → 상수항 1 없음

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

test_poly = poly.transform(test_input)  # 테스트 데이터도 같은 변환 적용

# ------------------- 선형 회귀 학습 -------------------

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)  # 다항 특성으로 학습
print(lr.score(train_poly, train_target))  # 훈련 세트 R^2 점수
print(lr.score(test_poly, test_target))    # 테스트 세트 R^2 점수

# ------------------- 다항 차수 5까지 확장 -------------------
poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)  # 특성 개수 확인 (차수 5라서 훨씬 많아짐)

lr.fit(train_poly, train_target)
print('5제곱 훈련 스코어')
print(lr.score(train_poly, train_target))  # 훈련 세트 성능 (거의 완벽히 맞출 수 있음 → 과적합 위험)

print('5제곱 테스트 스코어')
print(lr.score(test_poly, test_target))  # 테스트 세트 성능 (떨어질 수 있음 → 과적합 확인용)



