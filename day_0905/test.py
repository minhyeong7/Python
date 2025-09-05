# 전처리 (auto-mpg.csv) [horsepower]
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('data\\auto-mpg.csv')

df.columns=['mpg','cylinders','displacement','horsepower',
              'weight','acceleration','model year','origin',
              'name']


print(df.tail(20))
print()

#누락 데이터 확인하기
print("------누락데이터 확인하기----------")
df['horsepower'] = df['horsepower'].replace('?', np.nan)
print("널의 개수:",df['horsepower'].isnull().sum(axis=0))
print()

# 단위변환
print("------단위변환----------")
print(df['horsepower'].info())
print()
df['horsepower'] = df['horsepower'].astype('float')
print(df['horsepower'].info())
print()


# 누락 데이터 처리하기
print("------누락 데이터 처리하기----------")
df = df.dropna(subset=['horsepower'], axis=0)
df.info()
print()

# 필요없는 컬럼 삭제하기
print("------필요없는 컬럼 삭제하기----------")
df=df.drop('origin',axis=1)
print(df)
print()

# 범주 나눠보기, 범주별로 인코딩
print("------범주 나눠보기, 범주별로 인코딩----------")
count, bin_dividers =np.histogram(df['horsepower'],bins=[0,100,200,300])
print(bin_dividers)
print()

encoder = OneHotEncoder(sparse_output=False)
print(df.ndim)
print()

encoded = encoder.fit_transform(df)
print(encoded)
print()


# 중복행 확인(및 제거)
print("------중복행 확인(및 제거)----------")
df_dup_false=df['horsepower'].duplicated(keep='first')
print(df_dup_false)
print()

df4 = df['horsepower'].drop_duplicates(keep=False)
print(df4)
print()

# 데이터 스케일링(minmax,standard)
print("------데이터 스케일링(minmax,standard)----------")
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
df['horsepower_minmax'] = scaler.fit_transform(df[['horsepower']])
print(df['horsepower_minmax'])
print()



