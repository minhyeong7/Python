import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

titanic = sns.load_dataset('titanic')
df = titanic.loc[:,['age','sex','class','fare','survived']]

print(df.head())
print()

# class 로 그룹바이
# observed=True -- 데이터,실제 있는 조합을 만들겠다.
grouped = df.groupby(['class'],observed=True)

print(grouped) # 객체 반환

# 표준편차 집계
std_all = grouped.std(numeric_only=True)
print(std_all)
print()


# as_index=False 옵션 달아보기
std_all_index= df.groupby(['class'],observed=True, as_index=False).std(numeric_only=True)
print(std_all_index)
print()

# 그룹화 후 집계후, 컬럼 선택
print(std_all['fare'])
print()

# 그룹화 후, 컬럼 선택 후, 집계
std_fare=grouped['fare'].std(numeric_only=True)
print(std_fare)
print()

# 그룹별로 선택된 컬럼들의 집계결과
std_age_survived = grouped[['age', 'survived']].std(numeric_only=True)
print(std_age_survived)


# 그룹화 후, describe
print(grouped.describe())
print()

print(grouped[['class','sex']].value_counts())
print()

# 비교해보기1
print(df['class'].value_counts())
print()

# 비교해보기2
print(df[['class','sex']].value_counts())
print()

# ----------- 여러 함수나 컬럼별 다른 함수를 쓰기 위한 agg 메서드 ---------

# 그룹 객체에 aggregate() 메서드 적용
agg_mean = grouped.aggregate('mean',numeric_only=True)
print(agg_mean)
print()


# agg로도 쓸 수 있음
agg_mean2 = grouped.agg('mean',numeric_only=True)
print(agg_mean2)
print()

# 여러 집계함수를 적용
agg_all = grouped.agg(['min','max'])
print(agg_all)
print()


# 더 구체적으로 적용
agg_sep=grouped.agg({'fare':['min','max'],'age':'mean'})
print(agg_sep)
print()

def min_max(x):
    return x.max()-x.min()


# 각 그룹의 최대값과 최소값의 차이를 계산하여 그룹별로 집계
agg_minmax = grouped[['age','fare']].agg(min_max)
print(agg_minmax)
print()










