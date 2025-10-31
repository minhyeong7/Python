# IMDB 영화 리뷰 데이터셋 (단어가 이미 정수로 바꿔져있음.)
from keras.datasets import imdb

# 자주 사용하는 단어 200개만 사용 
# keras IMDB 단어사전이 이미 만들어져 있다.
# 그 단어 사전에 단어가 많지만, 200개만 취급을 하겠다.
(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=200)

print('\n훈련/테스트 데이터 크기')
print(train_input.shape, test_input.shape)
# (25000,) (25000,)
# 각각의 리뷰샘플이 파이썬리스트 '객체'로 이루어진 넘피 배열이다.

print('\n첫 번째 리뷰(객체) 길이 확인')
print(len(train_input[0])) # 218 (단어가 218개)

print('\n두 번째 리뷰(객체) 길이 확인')
print(len(train_input[1])) # 189 (단어가 189개)

print('\n첫 번째 리뷰 확인')
print(train_input[0]) # 어휘 사전(200개)에 없는 단어는 모두 2로 표시

print('\n타겟 데이터 20개 확인')
print(train_target[:20]) # 1 긍정 0 부정

# 훈련/검증 데이터 20% 분리
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

# 훈련세트 몇 가지 조사!

# 모든 리뷰의 길이를 조사해보자.
import numpy as np

# 샘플별로 길이 반환
lengths = np.array([len(x) for x in train_input])

print('\n샘플 길이 평균 / 중앙 값')
print(np.mean(lengths), np.median(lengths))
# 한쪽으로 치우쳐있다!

# 그래프로 확인 >>> 100 단어로 맞추면 적정.
import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

# 케라스에서 시퀀스 데이터 전처리 기능 제공
from keras.preprocessing.sequence import pad_sequences

# 100단어로, 길면 자르고 부족하면 0으로 패딩 (안정해주면 최대길이로 패딩)
train_seq = pad_sequences(train_input, maxlen=100)

print('\ntrain_seq.shape')
print(train_seq.shape)

print('\ntrain_seq 첫번째 샘플')
print(train_seq[0])

print('\ntrain_input 첫번째 샘플 마지막 부분')
print(train_input[0][-10:])
# 앞쪽을 잘라내고 뒷쪽을 살린 것을 알 수 있다.

print('\n다섯 번째 샘플')
print(train_seq[5])
# 앞쪽을 패딩하여 뒷쪽을 강조하였다.

# 검증 세트도 전처리.
val_seq = pad_sequences(val_input, maxlen=100)

# ---------- 순환 신경망 만들기 ----------

import keras 

model = keras.Sequential()
model.add(keras.layers.Input(shape=(100,200))) 
# 한샘플에 100 단어 / 한 단어에 200 원핫 인코딩 (단어사전 200개로 지정했기 때문)
model.add(keras.layers.SimpleRNN(8)) # 순환신경망 셀 8개
model.add(keras.layers.Dense(1, activation='sigmoid')) # 출력층

# 케라스 유틸 사용하여 원핫 인코딩
train_oh = keras.utils.to_categorical(train_seq)

print('\n원핫 인코딩 후 훈련셋 크기')
print(train_oh.shape) # (20000, 100, 200)

print('\n첫 번째 샘플, 첫 번째 단어 확인')
print(train_oh[0][0][:12]) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]

print('\n원소 200개 모두 합해서 1인지 확인')
print(np.sum(train_oh[0][0])) # 1

# 검증 세트도 원핫 인코딩
val_oh = keras.utils.to_categorical(val_seq)

# 모델 서머리
# 1672 = 200x8(웨이트) + 8x8(순환가중치) + 8(바이어스)
# 9 = 8(웨이트) + 1(바이어스)
print()
model.summary()

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
               validation_data=(val_oh, val_target),
               callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# -------------- 단어 임베딩 사용 --------------

# 원핫 인코딩은 샘플의 크기를 200차원까지 (너무) 늘렸다.
# 임베딩은 차원을 절약할 수 있다.
# 덕분에 단어사전은 500으로 늘려보자.
# (한 문장 단어수는 100으로 동일)

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

model_emb = keras.Sequential()
model_emb.add(keras.layers.Input(shape=(100,)))
model_emb.add(keras.layers.Embedding(500, 16)) # 단어사전 크기 500, 단어마다 차원 16 
# 임베딩 초기화 범위 [-0.05, 0.05]
model_emb.add(keras.layers.SimpleRNN(8))
model_emb.add(keras.layers.Dense(1, activation='sigmoid'))

print()
model_emb.summary()
# 8000 = 500x16 임베딩 파라미터
# 200 = 16x8(웨이트) + 8x8(순환웨이트) + 8(바이어스)

# 모델 컴파일
model_emb.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model_emb.fit(train_seq, train_target, epochs=100, batch_size=64,
               validation_data=(val_seq, val_target),
               callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()