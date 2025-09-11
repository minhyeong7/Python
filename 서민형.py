
print("==========================1번===========================")
# 1.  Seaborn의 titanic 데이터셋을 불러와 titanic 변수에 저장하시오.
import pandas as pd
import seaborn as sns


titanic=sns.load_dataset('titanic')
print("==========================2번===========================")
# 2.  Titanic 데이터의 기본 정보를 조회하시오
titanic.info()

print()
print("==========================3번===========================")
# 3.  Titanic 데이터의 행과 열 개수를 조회하고, 몇 차원 배열인지 조회하시오
print("행과 열:",titanic.shape)
print()
print("차원:",titanic.ndim)
print()

print("==========================4번===========================")
# 4.  첫 3행과 마지막 2행을 조회하시오
print(titanic.iloc[:3])
print()
print(titanic.iloc[:-3:-1])
print()
print("==========================5번===========================")
# 5.  loc을 사용해 첫 5행에서 열 ['survived','pclass','sex','age']만을 가진 데이터프레임 df_loc을 만들고, 출력하시오.
df_loc=titanic.loc[:5,'survived':'age']
print(df_loc)
print()
print("==========================6번===========================")
# 6.  iloc을 사용해 행 10~14(포함), 열 0~3(포함)을 추출해 df_iloc에 저장하고, 출력하시오.
df_iloc=titanic.iloc[10:15,0:4]
print(df_iloc)
print()
print("==========================7번===========================")
# 7.  원본을 훼손하지 않고(inplace=False) titanic에서 열 ['deck','embark_town']을 드랍한 새 데이터프레임 df_drop_cols를 만드시오.
df_drop_cols=titanic.drop(['deck','embark_town'],axis=1,inplace=False)
print(df_drop_cols)
print()
print("==========================8번===========================")
# 8.  결측치가 하나라도 있는 행을 드랍한 데이터프레임 df_dropna_rows를 만드시오.
df_dropna_rows=titanic.dropna()
df_dropna_rows.info()
print()
print("==========================9번===========================")
# 9.  각 열별 결측치 개수를 Series로 구하시오.
print(titanic.isnull().sum(axis=0))
print()

print("==========================10번===========================")
# 10.  age 열의 결측치 개수만 따로 출력하시오.
print(titanic['age'].isnull().sum(axis=0))
print()

print("==========================11번===========================")
# 11.  age 열의 평균값으로 해당 열의 결측치를 대체한 새로운 시리즈 age_filled를 만드시오(원본 불변).
mean=titanic['age'].mean()
age_filled=titanic['age'].fillna(mean)
print(age_filled)
print()

print("==========================12번===========================")
# 12.  대체 전후 age의 결측치 개수를 각각 출력하여 비교하시오.
print("대체 전:",titanic['age'].isnull().sum(axis=0),"대체 후:",age_filled.isnull().sum(axis=0))
print()

print("==========================13번===========================")
# 13.  embarked 열의 최빈값을 describe() 결과로 확인하시오.
most_freq2 = titanic['embark_town'].mode()
print(most_freq2.describe())

print("==========================14번===========================")
# 14.  그 최빈값으로 embarked의 결측치를 대체한 embarked_filled 시리즈를 만드시오(원본 불변).
most_freq = titanic['embarked'].value_counts(dropna=True).idxmax()
titanic['embarked']=titanic['embarked'].fillna(most_freq)
embarked_filled=titanic['embarked']=titanic['embarked'].fillna('S')
embarked_filled.info()

print("==========================15번===========================")
# 15.  수치형 열 중 ['age','fare']만 선택하여 0~1 범위로 Min-Max 스케일링한 데이터프레임 df_scaled를 만드시오(사전 결측 대체 필요 시 적절히 처리).
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scared = titanic.dropna(subset=['age','fare'], axis=0)
df_scared[['age','fare']] = scaler.fit_transform(df_scared[['age','fare']])
print(df_scared[['age','fare']])
print()

print("==========================16번===========================")
# 16.  스케일링 후 각 열의 최소/최대가 0과 1에 가깝게 되었는지 describe()로 확인하시오.
print(df_scared[['age','fare']].describe())
print()


print("==========================17번===========================")
# 17.  age를 기준으로 아동(018], 성인(18100]) 4구간으로 나누어 새 열 age_bin 을 생성하시오.
import numpy as np
count, bin_dividers = np.histogram(df_scared['age'], bins=[0,12,18,60,100])
names= ['아동', '청소년', '성인','노인']

df_scared['age_bin'] = pd.cut(x=titanic['age'],
                      bins=bin_dividers,
                      labels=names,
                      include_lowest=True)
print(df_scared)
print()

print("==========================18번===========================")
# 18.  각 구간별 인원수를 구하시오.
grouped = df_scared.groupby(['age_bin'],observed=True)
print(grouped['age_bin'].value_counts())
print()

print("==========================19번===========================")
# 19.  pclass와 sex로 그룹화하여 survived의 평균 생존율을 구하시오.
grouped = titanic.groupby(['pclass','sex'],observed=True)
agg_all = grouped.agg({'survived':['mean']})
print(agg_all)
print()


print("==========================20번===========================")
# 20.  위 결과를 생존율 내림차순으로 정렬하시오.
print()

print("==========================21번===========================")
# 21.  age_bin(문항17)과 sex로 그룹화하여 fare의 중앙값을 구하시오
grouped2 = df_scared.groupby(['age_bin','fare'],observed=True)
median = grouped.agg({'fare':['median']})
print(median)


print()

print(titanic)
# 11.  age 열의 평균값으로 해당 열의 결측치를 대체한 새로운 시리즈 age_filled를 만드시오(원본 불변)
corr = titanic.corr(numeric_only=True)
print(corr)

titanic['age_filled'] = titanic['age'].fillna(
    titanic.groupby('pclass')['age'].transform('mean')
)

print(titanic[['fare', 'age', 'age_filled']].head(20))



# age_filled=titanic['age'].fillna(mean)
# print(age_filled)
# print()

