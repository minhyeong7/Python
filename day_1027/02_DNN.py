import keras

print('\n----- 데이터셋 로드 -----')
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 한 층을 더 추가 해보자.
inputs = keras.layers.Input(shape=(784,)) # 입력층
dense1 = keras.layers.Dense(100, activation='sigmoid') # 은닉층 (입력차원의 1/2 ~ 1/4)
# a x 4 + 2 = b  >>>  b x 3 - 5 = c  >>> c = a x 12 + 1 (결국 선형 결합)
# 비선형성, 복잡성을 주기위해 활성화 함수 통과 ex)sigmoid
dense2 = keras.layers.Dense(10, activation='softmax') # 출력층
model = keras.Sequential([inputs, dense1, dense2])
print()
model.summary()


# ----- 다른 방법으로 층 추가 1 -----

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(100, activation='sigmoid', name='은닉층'),
    keras.layers.Dense(10, activation='softmax', name='출력층')
], name='패션 MNIST 모델')
print()
model.summary()
# 784 x 100 + 100 = 78500
# 100 x 10 + 10 = 1010

# ----- 다른 방법으로 층 추가 2 -----
# pip install keras
model = keras.Sequential()
model.add(keras.layers.Input(shape=(784,)))
model.add(keras.layers.Dense(100, activation='sigmoid')) # 은닉층 (활성화함수 = 시그모이드)
model.add(keras.layers.Dense(10, activation='softmax')) # 출력층 (다중분류)
print()
model.summary()

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5)
# 디폴트 배치 = 32개 (32개 데이터마다 파라미터 1번 학습)

# ---------- 렐루 활성화 함수 ----------
# 시그모이드 함수는 미분 값이 너무 작아서 입력층에 가까워질 수록 파라미터 업데이트가 거의 안된다.
# "기울기 소실" 문제

model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28))) # 2D 그림 그대로 입력
model.add(keras.layers.Flatten()) # 플래튼 층에서 자동으로 펼쳐서 입력
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax')) # 출력층 
print()
model.summary()

# reshape 없이 바로 적용해보기
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0 # 0~1 사이 숫자로 스케일링
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5) # 모델 훈련 

print('\n----- 모델 평가 (relu) -----')
print(model.evaluate(val_scaled, val_target))


# ----- 옵티마이저 ----- (파라미터 업데이트 방식)

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 위와 똑같은 코드
sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습률 조정
sgd = keras.optimizers.SGD(learning_rate=0.1)

# 모멘텀 - 이전 단계의 방향을 일정 비율로 유지하여 진동을 줄이고 빠르게 수렴하도록
# 이전 속도(관성)를 더해서 가자
sgd = keras.optimizers.SGD(momentum=0.9)

# 모멘텀 + 네스테로프 모멘텀
# 더 정확한 방향으로 이동 (한스탭을 미리 보고 이동)
# 모멘텀 보다 빠르고 안정적인 수렴
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)

# adagrad - 각 파라미터별로 학습률을 다르게 조정함
# 자주 업데이트 되는 파라미터는 학습률을 작게, 드물게 업데이트 되는 파라미터는 학습률 크게
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# RMSProp - adagrad의 학습률 감소 문제 해결 (학습률이 너무 빨리 줄어드는 문제 해결)
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Adam - 모멘텀 + RMSProp의 장점을 결합
# 기본적으로 가장 많이 쓰이는 옵티마이저
# 빠른수렴 + 자동 학습률 조정
adam = keras.optimizers.Adam()

model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5)

print('\n----- 모델 평가 adam -----')
print(model.evaluate(val_scaled, val_target))