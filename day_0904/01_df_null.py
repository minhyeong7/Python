import seaborn as sns
import pandas as pd

pd.set_option('display.unicode.east_asian_width', True)  # 한글·중국어·일본어 같은 동아시아 문자가 표에서 칸 맞춰 정렬되도록 설정
pd.set_option('display.max_columns', None)              # DataFrame 출력 시 모든 열을 표시 (기본은 일정 개수까지만 표시됨)
pd.set_option('display.max_rows', 100)                  # DataFrame 출력 시 최대 100행까지만 표시
pd.set_option('display.width', 300)                     # 출력 가로폭을 300으로 설정 (기본보다 넓게 → 줄바꿈 덜 생김)



df = sns.load_dataset('titanic')
print(df)

print(df.head(10))
print()

df.info()
print()
print("----------- 누락 데이터 확인 -----------")

# deck 열의 NaN 개수 계산하기
print(df['deck'].value_counts())
print()

# isnull() 메서드로 누락 데이터 찾기
print(df.head().isnull())
print()

# notnull() 메서드 
print(df.head().notnull())
print()


# isnull() 메서드로 누락 데이터 개수 구하기
print(df.isnull().sum(axis=0))
print()

import missingno as msno 
import matplotlib.pyplot as plt

# 매트릭스 그래프
# plt.figure(figsize=(12,6))
# msno.matrix(df)
# plt.show()

# 막대 그래프
# msno.bar(df)
# plt.show()

# 히트맵 (누락 데이터 변수 간 상관관계)
# msno.heatmap(df)
# plt.show()

# msno.dendrogram(df)
# plt.show()

# 덴드로그램
# msno.dendrogram(df)
# plt.show()

### ------ 누락 데이터 표현 ------

# 기존 방식(np.nan): 정수형 자료가 float로 변환됨
ser1 = pd.Series([1,2,None])
print(ser1)
print()

# 정수형이 그대로 유지됨. (결측치는 pd.NA로 표현) 
ser2 = pd.Series([1,2,None], dtype='Int64')
print(ser2)


