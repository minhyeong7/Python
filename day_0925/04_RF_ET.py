import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# ------------------- 데이터 불러오기 -------------------
data = fetch_california_housing()                       # 캘리포니아 주택 가격 데이터셋 불러오기
X = pd.DataFrame(data.data, columns=data.feature_names) # 입력 변수들을 DataFrame으로 변환 (8개의 특성)
y = pd.Series(data.target)                              # 타겟 변수(집값)를 Series로 변환

print(X.head())    # 입력 데이터 상위 5개 행 확인
X.info()           # 데이터 타입, 결측치 여부, 행/열 수 확인
print(y.head())    # 타겟 데이터 상위 5개 확인

# ------------------- 훈련/테스트 데이터 분할 -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # 80% 훈련, 20% 테스트 분할, random_state=42로 재현 가능하게 고정

# ------------------- 모델 정의 -------------------
rf_model = RandomForestRegressor(random_state=42)  # 랜덤포레스트 회귀 모델, 앙상블 기반
et_model = ExtraTreesRegressor(random_state=42)    # 엑스트라트리 회귀 모델, 랜덤포레스트와 유사하지만 분할 무작위성 ↑

# ------------------- 하이퍼파라미터 그리드 설정 -------------------
param_grid = {
    'n_estimators': [50, 100],     # 트리 개수, 많을수록 안정적이지만 학습 시간 ↑
    'max_depth': [None, 10, 20],   # 트리 최대 깊이, None이면 제한 없음, 깊을수록 과적합 위험
    'min_samples_split': [2, 5]    # 노드 분할 최소 샘플 수, 작으면 세밀하게 분할 → 과적합 가능
}

# ------------------- 그리드 서치 설정 -------------------
rf_grid = GridSearchCV(
    rf_model,          # 탐색할 모델
    param_grid,        # 하이퍼파라미터 후보
    n_jobs=-1,         # 모든 CPU 코어를 사용해서 탐색 병렬 처리
    verbose=1          # 진행 상황 출력, 1=간단, 3=자세히
)
et_grid = GridSearchCV(
    et_model,
    param_grid,
    n_jobs=-1,
    verbose=1
)

# ------------------- 모델 학습 (그리드 서치 실행) -------------------
print('랜덤포레스트 그리드 서치')
rf_grid.fit(X_train, y_train)   # 훈련 데이터로 모든 하이퍼파라미터 조합 학습 + 교차검증

print()

print('엑스트라트리 그리드 서치')
et_grid.fit(X_train, y_train)   # 훈련 데이터로 모든 하이퍼파라미터 조합 학습 + 교차검증
print()

# ------------------- 최적 하이퍼파라미터 출력 -------------------
print('랜덤포레스트 best: ', rf_grid.best_params_)  # GridSearchCV가 선택한 최적의 파라미터
print('엑스트라트리 best: ', et_grid.best_params_)

# ------------------- 최적 모델 추출 -------------------
rf_best = rf_grid.best_estimator_  # 최적 파라미터로 학습된 랜덤포레스트 모델
et_best = et_grid.best_estimator_  # 최적 파라미터로 학습된 엑스트라트리 모델

# ------------------- 성능 평가 -------------------
print('랜덤포레스트 스코어: ', rf_best.score(X_test, y_test))  # 테스트 세트에서 R² 결정계수 출력
print('엑스트라트리 스코어: ', et_best.score(X_test, y_test))  # 테스트 세트에서 R² 결정계수 출력

# ✅ 주의:
# 1. score()는 회귀에서는 R² 값(0~1)으로, 1에 가까울수록 예측이 정확
# 2. GridSearchCV 내부적으로 교차검증 수행 → 하이퍼파라미터 튜닝 후 최적 모델 선택
# 3. 테스트 데이터는 최종 평가용, 훈련/검증 과정에서는 사용하지 않음
