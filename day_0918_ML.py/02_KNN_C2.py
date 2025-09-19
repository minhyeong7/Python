fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 길이와 무게 데이터를 묶어서 2차원 리스트로 구성
fish_data = [[l,w] for l,w in zip(fish_length, fish_weight)]
# 도미(1) 35개 + 방어(0) 14개 라벨 생성
fish_target = [1]*35 + [0]*14

print("-----훈련 데이터-----")
print(fish_data)   # (길이, 무게) 데이터 확인
print('-----타겟 데이터------')
print(fish_target) # 각 데이터의 라벨 확인

# 훈련셋 (샘플링 편향) 잘못학습
# 앞 35개는 모두 도미 
train_input = fish_data[:35]   # 앞 35개 → 도미만 포함
train_target = fish_target[:35]

# 테스트셋 # 뒤 14개는 모두 방어 
test_input = fish_data[35:]   # 뒤 14개 → 방어만 포함
test_target = fish_target[35:]

from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier()

# 학습
kn.fit(train_input, train_target)  # 훈련 데이터로 학습
print("-----학습 완료-----")
print()

print('-----테스트 스코어-----')
print(kn.score(test_input,test_target))  # 편향된 데이터 때문에 정확도 0.0
print()


# 넘파이 조금 배우고 가보자!
# 파이썬의 대표적인 배열 라이브러리! (고차원을 리스트보다 손쉽게 표현)
# C 기반이라서 속도가 리스트보다 훨씬빠르다!
import numpy as np
input_arr = np.array(fish_data)     # 리스트를 넘파이 배열로 변환 → 2차원 배열
target_arr = np.array(fish_target)  # 라벨 리스트도 넘파이 배열로 변환

print('-----넘파이 배열 출력------')
print(input_arr)      # (길이, 무게) 전체 출력
print('----input_arr.shape------')
print(input_arr.shape)  # (49, 2) → 샘플 49개, 특성 2개(길이, 무게)
print()

np.random.seed(42) 
index=np.arange(49) # 49개의 숫자 만들기 (0~48까지 인덱스)
print('섞기전:',index)
np.random.shuffle(index)  # 랜덤하게 섞기
print('섞은후:',index)
np.random.shuffle(index)  # 한 번 더 섞기 (시드 고정이므로 여전히 랜덤하지만 일관됨)
print('두번 섞은후:',index)

# 넘파이 인덱싱 예시
print(input_arr[[1,3]])  # 1번째, 3번째 데이터를 뽑아서 출력 (Fancy indexing)
print()

# 섞어 담기
train_input=input_arr[index[:35]]   # 섞인 인덱스 중 앞 35개 → 훈련셋
train_target=target_arr[index[:35]] # 같은 순서로 타겟도 섞어서 맞춤

test_input= input_arr[index[35:]]   # 섞인 인덱스 중 뒤 14개 → 테스트셋
test_target=target_arr[index[35:]]  # 타겟도 같은 순서로 맞춤

print('-----섞은후, 분할 후 확인-----')
print(input_arr[29],train_input[0]) # 29번째 데이터가 실제로 훈련셋 첫 번째로 들어갔는지 확인
print()

print('-----훈련 인풋-----')
print(train_input)   # 훈련 입력값 전체 확인

import matplotlib.pyplot as plt

# 훈련 데이터와 테스트 데이터 시각화 (색으로 나눠 보기 위함)
plt.scatter(train_input[:,0], train_input[:,1]) # 훈련 데이터 (파란색)
plt.scatter(test_input[:,0], test_input[:,1])   # 테스트 데이터 (주황색)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 및 결과
kn.fit(train_input,train_target)           # 섞인 훈련 데이터로 학습
print(kn.score(test_input,test_target))    # 테스트 정확도 확인 (편향 해결 후 정상적으로 나옴)
print("-----테스트 예측------")
print(kn.predict(test_input))              # 테스트셋 예측 결과
print("-----정답지-------")
print(test_target)                         # 실제 라벨
print()

'''
도미 35 / 방어 14 데이터를 준비
훈련 세트 35개 테스트세트 14 나눴다
그런데 훈련에는 다 도미 / 테스트는 다 방어로 나눠서 >> 훈련 잘못됨.

(이참에 넘피 맛보기) 
넘피 시드 고정
넘피로 훈련세트와 테스트세트를 골고루 섞어 잘 나눴다. (35 : 14)
** 그래프로 잘 나누었는지 확인
다시 테스트 >>> 성공!

** 스코어 확인
** 예측값 / 정답 대조
'''



