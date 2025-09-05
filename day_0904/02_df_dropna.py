import pandas as pd
import seaborn as sns
pd.set_option('display.max_rows', 100)


df = sns.load_dataset('titanic')

print(df.isnull().sum(axis=0))
print(df['age'].isnull().sum(axis=0))
print()

print("--------- 누락데이터 제거 ----------")

print(df)

# 데이터에 널 값이 하나라도 있는 사람(행) 데이터는 다 날아감
df_dropna = df.dropna() # default axis = 0
df_dropna.info()
print()

# 데이터에 널 값이 하나라도 있는 컬럼(열)은 다 날아감
df_dropna2 = df.dropna(axis=1) 
df_dropna2.info() 
print()

# 유효한 데이터 500개 이상은 되어야 살아남음
df_dropna3 = df.dropna(axis=1, thresh=500) 
df_dropna2.info() 
print()

# age가 널값인 행만 지워라
df_age = df.dropna(subset=['age'], axis=0)
df_age.info()
print()

# age, deck 중에 하나라도 널값이 있으면 지워라
df_age_deck = df.dropna(subset=['age','deck'], axis=0) # default how='any'
df_age_deck.info()
print()


# age, deck 모두 널값이 있으면 지워라
df_age_deck = df.dropna(subset=['age','deck'],how='all', axis=0)
df_age_deck.info()
print()

print("--------- age 널 값을 age 평균값으로 채우기 ----------")
print()

age_mean=df['age'].mean()

df['age']=df['age'].fillna(age_mean)
print(df['age'].isnull().sum(axis=0))

print("---------- embark_town (최빈값으로 대체) ---------")
print()

# 숫자형의 산술정보
print(df.describe(include=object))
print()

# 문자형의 통계정보
em_freq = df['embark_town'].value_counts(dropna=True)
print(em_freq)
print()

# embark_town의 고윳값별 카운트
most_freq = df['embark_town'].value_counts(dropna=True).idxmax()
print(most_freq)
print()

# embark_town의 최빈값
# .mode() 는 시리즈의 최빈값을 시리즈로 반환
most_freq2 = df['embark_town'].mode()[0]
print(most_freq2)
print()

df.info()
print("--------------------------------")
print(df.loc[825:831,'embark_town'])
print(df['embark_town'][820:831])

# embark_town 열의 NaN 값을 최빈값으로 채워넣기
df['embark_town']=df['embark_town'].fillna(most_freq)
df['embarked']=df['embarked'].fillna('S')
df.info()

print("---------- 근처 값으로 대체------------")
print()

df=sns.load_dataset('titanic')

# 데이터프레임 복제하기
df2= df.copy()

print(df['embark_town'][825:831])
print()

# 이전행 (828행) 값으로 채워라 
df['embark_town'] = df['embark_town'].ffill()
print(df['embark_town'][825:831])
print()


# 이후행 (830행) 값으로 채워라 
df2['embark_town'] = df2['embark_town'].bfill()
print(df2['embark_town'][825:831])
print()

