# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

import matplotlib.pyplot as plt

# 도미 길이와 무게 산점도
plt.scatter(bream_length,bream_weight)
plt.xlabel('length')  # x축 이름
plt.ylabel('weight')  # y축 이름
plt.show()

# 방어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 방어 길이와 무게 산점도
plt.scatter(smelt_length,smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 도미+방어 데이터 합치기
length = bream_length + smelt_length
weight= bream_weight+ smelt_weight

# 인풋데이터: [길이, 무게] 형태로 리스트 생성
fish_data = [[l,w] for l,w in zip(length,weight)]
print(fish_data)
print()

# 타겟 데이터: 도미=1, 방어=0
fish_target =[1]*35 + [0]*14
print(fish_target)

# k-최근접 이웃 알고리즘 학습
# KNN(K-Nearest Neighbors)
# k- 근처 몇개의 이웃을 참고하지 (K 개)
from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier()

# KNN 분류 학습
kn.fit(fish_data,fish_target)
print('kn 모델 변수에 학습 완료')
print()

# 방어와 도미 그래프 + 예측 포인트
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length,smelt_weight)
plt.scatter(30,600,marker='^')  # 예측하고 싶은 점을 삼각형으로 표시
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()  # 그래프 출력 주석 처리 (원하면 활성화 가능)

# 예측 결과 확인
print('예측 결과')
print(kn.predict([[30,600]]))  # 길이30, 무게600인 물고기 예측
print()

# KNN - 모든 점의 거리를 계산해서 가장 가까운 5개의 이웃을 보고 분류
# 모든 데이터의 정보를 가지고 있음
# 사실상 훈련이라고 하기엔 모호함
print(kn._fit_X)  # 학습 데이터 확인
print()
print(kn._y)      # 학습 라벨 확인
print()

# 디폴트 이웃 수 5개
# k를 49개로 바꿔 보면??
kn49= KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data,fish_target)

# 분류 모델에서 score는 맞춘 수 / 전체
print(kn49.score(fish_data,fish_target))  # 학습 데이터 기준 정확도

# 5개부터 49개 까지

# 이웃 수(k)를 5부터 49까지 바꿔가며 반복
for i in range(5,50):
    # i개의 이웃을 사용하는 KNN 분류기 생성
    kn = KNeighborsClassifier(n_neighbors=i)
    
    # fish_data(입력 특징), fish_target(정답 레이블)으로 학습
    kn.fit(fish_data, fish_target)
    
    # 학습된 모델의 정확도 계산 (전체 데이터 기준)
    num = kn.score(fish_data,fish_target)
    
    # 정확도가 1보다 작아지는 첫 번째 순간에 출력하고 반복 종료
    if num < 1:
        print(i)     # 현재 이웃 수(k)
        print(num)   # 해당 k에서의 정확도
        break

# 여러 물고기 예측
print('아무 생선 3마리 예측 시켜보기')
print(kn49.predict([[30,600],[20,100],[15,70]]))
print()

# 학습 데이터 준비(인풋데이터, 타겟데이터)
# 모델 선정 및 불러오기
# 데이터로 모델 학습
# 스코어 확인/ 특정데이터예측
# (그래프로 확인)

# 5개부터 시작해서 스코어 확인
# 과연 몇일 때 스코어가 1미만이 될까? 알아내기
