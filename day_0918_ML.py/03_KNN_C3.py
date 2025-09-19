fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]  # 물고기 길이 데이터


fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]  # 물고기 무게 데이터


import numpy as np

# 넘피 컬럼스택(리스트들을 열 단위로 묶음 → 2차원 배열로 변환)
print(np.column_stack(([1,2,3],[4,5,6])))  
print()

# 도미/방어의 길이와 무게를 열 단위로 묶어 fish_data 생성
fish_data = np.column_stack((fish_length, fish_weight))  
print(fish_data[:5])  # 앞 5개 샘플 출력
print()

# 넘피 배열 맛보기 (1, 0, 특정값 채우기)
print(np.ones(5))       # 크기 5짜리 1로 채워진 배열
print(np.zeros(5))      # 크기 5짜리 0으로 채워진 배열
print(np.full(5,2))     # 크기 5짜리 2로 채워진 배열
print(np.full((3,3),7)) # 3x3 크기 배열을 7로 채움

# 타겟 데이터 만들기 (도미=1, 35개 / 방어=0, 14개)
fish_target = np.concatenate((np.ones(35), np.zeros(14)))  
print(fish_target)  
print()

from sklearn.model_selection import train_test_split

# 데이터를 알아서 섞고 나눔 (기본 비율은 train: 75%, test: 25%)
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, test_size=0.25, random_state=42  # test_size=0.25 → 25%는 테스트, 시드 고정
)

print(train_input.shape, test_input.shape)  # 훈련/테스트 데이터 크기 확인
print()
print(train_target.shape, test_target.shape)  # 훈련/테스트 타겟 크기 확인
print()
print(test_target)  # 실제 테스트 타겟 값 확인
print()

# KNN 분류기 불러오기
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()  
kn.fit(train_input, train_target)  # 훈련 데이터로 모델 학습

print('-----훈련 스코어------')
print(kn.score(test_input, test_target))  # 테스트 데이터 정확도 출력
print()

# 새로운 샘플 (25cm, 150g) → 월척 데이터
import matplotlib.pyplot as plt

plt.scatter(train_input[:,0], train_input[:,1])  # 훈련 데이터 산점도
plt.scatter(25,150, marker='^')  # 새로운 샘플 '^' 삼각형 표시
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print('-----월척의 종류는???------')
print(kn.predict([[25,150]]))  # 새로운 데이터 예측
print()

# 새로운 샘플 근처 이웃 확인
distances, indexes = kn.kneighbors([[25,150]])  
print('근처 이웃 인덱스', indexes)  
print()

# 근처 이웃 시각화
plt.scatter(train_input[:,0], train_input[:,1])  # 전체 데이터
plt.scatter(25,150, marker='^')  # 새로운 샘플
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')  # 이웃들 'D' 표시
plt.xlabel('length')
plt.ylabel('weight')

# x축 범위를 넓혀서 확인 (일부 데이터가 뭉쳐있어서 압축됨)
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0,1000))  # x축 범위 0~1000으로 확장
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 스케일링 (데이터 표준화) → 평균 0, 표준편차 1로 맞춤
mean = np.mean(train_input, axis=0)  # 각 컬럼 평균
std = np.std(train_input, axis=0)    # 각 컬럼 표준편차

print("평균:", mean)
print("표준편차:", std)

train_scaled = (train_input - mean) / std  # 훈련 데이터를 표준화

plt.scatter(train_scaled[:,0], train_scaled[:,1])  # 스케일링된 데이터 시각화
plt.scatter(25,150, marker='^')  # 아직 스케일링 안 한 월척
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 월척도 스케일링 해줘야 올바르게 비교 가능
new = ([25,150] - mean) / std  # 새로운 데이터도 동일하게 표준화

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')  # 표준화된 월척 표시
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 표준화된 데이터로 다시 학습
kn.fit(train_scaled, train_target)

# 테스트 데이터도 반드시 같은 mean, std로 스케일링해야 함
test_scaled = (test_input - mean) / std  

print('스코어:', kn.score(test_scaled, test_target))  # 표준화 후 정확도
print('월척은? :', kn.predict([new]))  # 표준화된 월척 예측

# 월척 근처 이웃들 확인
distances, indexes = kn.kneighbors([new])  

# 이웃들 시각화
plt.scatter(train_scaled[:,0], train_scaled[:,1])  
plt.scatter(new[0], new[1], marker='^')  
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')  
plt.xlabel('length')
plt.ylabel('weight')

plt.show()
