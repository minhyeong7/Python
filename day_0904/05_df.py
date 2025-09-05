import pandas as pd
import numpy as np


pd.set_option('display.unicode.east_asian_width', True)  # 한글·중국어·일본어 같은 동아시아 문자가 표에서 칸 맞춰 정렬되도록 설정
pd.set_option('display.max_columns', None)              # DataFrame 출력 시 모든 열을 표시 (기본은 일정 개수까지만 표시됨)
pd.set_option('display.max_rows', 100)                  # DataFrame 출력 시 최대 100행까지만 표시
pd.set_option('display.width', 300)      

df = pd.read_csv('./data/auto-mpg.csv',header=None)

df.columns=['mpg','cylinders','displacement','horsepower',
              'weight','acceleration','model year','origin',
              'name']
# horsepower 열의 누락데이터('?') nan으로 바꾼 뒤 drop 한 다음, 
# horsepower 열의 타입을 float 으로 변환!!

print(df.tail(30))
print(df['horsepower'].sample(3))
print()
df['horsepower'] = df['horsepower'].replace('?', np.nan)
df = df.dropna(subset=['horsepower'],axis=0)
df['horsepower'] = df['horsepower'].astype('float')
print(df['horsepower'].sample(3))
print()


print("--------- horsepower 구간 나누기 ----------")
# 각 구간에 속하는 데이터 개수(count), 경계값 리스트(bin_dividers) 반환
# ex) bins = 3 으로 하면 세 구간으로 균등 분할
count, bin_dividers =np.histogram(df['horsepower'],bins=[0,100,200,300])
print(bin_dividers)
print()
print(count)
print()
print(df.describe())

bin_names=['저출력','보통출력','고출력']

# pd.cut 함수로 각 데이터를 3개의 bin에 할당
df['hp_bin'] = pd.cut(x=df['horsepower'],
                      bins=bin_dividers,
                      labels=bin_names,
                      include_lowest=True
                      )
print(df)

print()

# ----------- 더미 변수 -----------

# hp_bin 컬럼의 범주형 데이터를 더미 변수로 변환 (float)
horsepower_dumies = pd.get_dummies(df['hp_bin'],dtype=float)
print(horsepower_dumies.head(15))
print(type(horsepower_dumies))
print()

horsepower_dumies_drop =pd.get_dummies(df['hp_bin'],dtype=float, drop_first=True)


print(horsepower_dumies_drop.head())


# -------------------------------------------------------------------

from sklearn import preprocessing #  pip install scikit-learn

label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder(sparse_output=False)

# 사전순으로 0,1,2... 배정
# label encoder로 문자열 범주를 숫자형 범주로 변환
onehot_labeled = label_encoder.fit_transform(df['hp_bin'].head(15))
print(onehot_labeled)
print(type(onehot_labeled))
print()

# 2차원 행렬로 형태 변경
onehot_reshaped = onehot_labeled.reshape(len(onehot_labeled),1)
print(onehot_reshaped)
print(type(onehot_reshaped))
print()


onehot_fitted = onehot_encoder.fit_transform(onehot_reshaped)
print(onehot_fitted) # 희소행렬 반환
print()
# print(onehot_fitted.toarray())
print()
# OneHotEncoder(sparse_output=False) 옵션을 주면 바로 어레이로 반환


encoded_df = pd.DataFrame(onehot_fitted.toarray(), columns=onehot_encoder.get_feature_names_out(df[['hp_bin']].columns))
print(encoded_df)
print()





