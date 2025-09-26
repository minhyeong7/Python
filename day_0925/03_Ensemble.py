# 앙상블 (두명 이상이 함께 연주하는)
# Random Forest (랜덤포레스트)
# 100개의 결정트리 -- 분류(다수결) / 예측(평균)
# 부트스랩 샘플 (중복허용 랜덤 추출)
# 특성선택 - 랜덤선택 (총 특성개수의 제곱근 개수만큼 무작위) - 분류일때만 (회귀x)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

train_input, test_input, train_target, test_target =  train_test_split(
    data, target, test_size=0.2, random_state=42)

print('인풋데이터 개수')
print(train_input.shape)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# 100그루의 나무
# 5197개의 데이터에서 각각 나무들은 5197개의 랜덤(중복) 데이터를 뽑음.
# 4:1 비율로 교차:검증 세트로 나눈뒤, 그 훈련데이터에서 다시 한 번 부트스트랩 샘플링
# return_train_score=True --- 검증점수 뿐만 아니라 훈련 점수도 반환
rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

# 5번의 교차검증에 대한 훈련/검증 스코어 평균
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

rf.fit(train_input, train_target)
print('랜덤포레스트 특성 중요도')  # 과대적합을 줄이고 일반화 성능을 높인다.
print(rf.feature_importances_) 

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(train_input, train_target)

print('결정나무 특성 중요도')
print(dt.feature_importances_)


# 특별한 기능 oob_score
# 부트스트랩에서 뽑히지 않은 샘플들로 자체 테스트 진행!
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print('oob 스코어 결과')
print(rf.oob_score_)


# ----------------- 엑스트라 트리 -----------------
# 나무가 기본 100개
# 데이터는 전체 샘플사용 (옵션으로 부트스트랩 샘플링 가능)
# 특성선택은 랜덤포레스트와 같음 (랜덤선택)
### 무작위 노드분할 (결정나무의 splitter=random 옵션과 같음)
### >>>> 과대적합을 막고, 속도를 올린다.

from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
score = cross_validate(et, train_input, train_target,
                       return_train_score=True, n_jobs=-1)

print('엑스트라 트리 훈련/검증 스코어 평균')
print(np.mean(score['train_score']), np.mean(score['test_score']))

et.fit(train_input, train_target)
print('엑스트라 트리 특성 중요도')
print(et.feature_importances_)