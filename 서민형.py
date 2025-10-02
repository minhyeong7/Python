import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


print("------------------데이터 전처리------------------------------")
# 데이터 불러오기
df = sns.load_dataset('titanic')

# 널값 제거
df = df.dropna(subset=['age','fare'])

# 필요한 컬럼 선택 및 범주형 숫자 변환
df = df[['survived','pclass','sex','age','fare']]
df['sex'] = df['sex'].map({'male':1,'female':0})

# 특성과 타겟 분리
X = df.drop('survived', axis=1)
y = df['survived']

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("---------------데이터 간단 시각화------------------------")

# 생존자 수
sns.countplot(x='survived', data=df)
plt.title('생존자 수'); plt.show()

# 성별에 따른 생존자 수
sns.countplot(x='survived', hue='sex', data=df)
plt.title('성별에 따른 생존자 수'); plt.show()

# 객실 등급별 생존자 수
sns.countplot(x='survived', hue='pclass', data=df)
plt.title('객실 등급별 생존자 수'); plt.show()

# 나이 분포별 생존자
sns.histplot(data=df, x='age', hue='survived', bins=30, kde=False)
plt.title('나이 분포별 생존자'); plt.show()

print("-----------의사결정모델----------------------------------")
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay

# 그리드 서치로 최적 하이퍼파라미터 찾기
dt = DecisionTreeClassifier(random_state=42)
param_grid = {'max_depth':[2,3,4,5,6], 'min_samples_split':[2,5,10], 'min_samples_leaf':[1,2,4]}
grid = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', verbose=1)
grid.fit(X_train, y_train)

best_dt = grid.best_estimator_
print("최적 하이퍼파라미터:", grid.best_params_)

# ROC AUC와 정확도
y_pred_proba = best_dt.predict_proba(X_test)[:,1]
auc_dt = roc_auc_score(y_test, y_pred_proba)
acc_dt = best_dt.score(X_test, y_test)
print(f'의사결정나무 ROC AUC: {auc_dt:.3f}, 정확도: {acc_dt:.3f}')

# ROC 곡선
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_dt, estimator_name='Decision Tree').plot()
plt.title('ROC Curve - 의사결정나무'); plt.show()

# 트리 시각화
plt.figure(figsize=(15,10))
plot_tree(best_dt, filled=True, feature_names=X.columns, class_names=['죽음','생존'], rounded=True, fontsize=12)
plt.title('의사결정나무 시각화'); plt.show()

print("--------------로지스틱 회귀 모델--------------------------------")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from scipy.special import expit
import numpy as np

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# ROC AUC와 정확도
y_lr_proba = lr.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_lr_proba)
acc_lr = lr.score(X_test, y_test)
print(f'로지스틱 회귀 ROC AUC: {auc_lr:.3f}, 정확도: {acc_lr:.3f}')

# ROC 곡선
RocCurveDisplay.from_predictions(y_test, y_lr_proba, name='Logistic Regression')
plt.title('ROC Curve - 로지스틱 회귀'); plt.show()

# 테스트 데이터 상위 5개 샘플 z 값과 시그모이드 확률 확인
X_sample = X_test[:5]
z_values = lr.decision_function(X_sample)
sigmoid_probs = expit(z_values)
pred_classes = lr.predict(X_sample)
print("상위 5개 샘플 z값:", z_values)
print("시그모이드 확률:", sigmoid_probs)
print("예측 클래스:", pred_classes)
print("모델 클래스:", lr.classes_)
print("모델 파라미터 (가중치/절편):", lr.coef_, lr.intercept_)


print("-----------비교/평가---------------")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 의사결정나무 확률
y_dt_proba = best_dt.predict_proba(X_test)[:,1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_dt_proba)
auc_dt = roc_auc_score(y_test, y_dt_proba)

# 로지스틱 회귀 확률
y_lr_proba = lr.predict_proba(X_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_lr_proba)
auc_lr = roc_auc_score(y_test, y_lr_proba)

# ROC 곡선 그리기
plt.figure(figsize=(8,6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc_dt:.3f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()
