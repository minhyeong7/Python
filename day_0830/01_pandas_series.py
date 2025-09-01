# 터미널에 pip install pandas
import pandas as pd

# 시리즈 = 인덱스와 데이터
print()

print("------리스트를 시리즈로-----")

list_data=['a',2,'b']

sr=pd.Series(list_data)
print(sr)
print(sr.index) # dtype: object
print(sr.values) #['a' 2 'b']
print(type(sr)) # <class 'pandas.core.series.Series'> 넘피 기반 구조인 판다스는 데이터 전체를 object로 가진다
print(sr.dtype) # object
print(len(sr)) # 3
print(sr.shape) #(3,) → 1차원 배열, 길이 3
print(sr.ndim) #1 배열의 차원 수


print()
print("-----튜플을 시리즈로-----")

tup_data=('영민','남',True)

tup_data=('영민','남',True)
sr2= pd.Series(tup_data,index=['이름','성별','학생여부'])
print(sr2)
print(sr2[0])
print(sr2['이름'])
print(sr2[[1,2]])
print(sr2[1:2])
print(sr2[0:2])
print(sr2.index)

print()
print("-----시리즈 생성-----")
print(pd.Series()) # Series([], dtype: object)
print(pd.Series(5)) 
print(pd.Series(5,[])) # Series([], dtype: int64)



