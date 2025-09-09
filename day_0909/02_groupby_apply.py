import pandas as pd
import seaborn as sns


titanic=sns.load_dataset('titanic')
df= titanic[['age','sex','class','fare','survived']]
print(df)
print()


# class로 groupby
grouped = df.groupby(['class'],observed=True)

# 각 그룹별 요약 통계정보 집계
agg_grouped = grouped[['age','survived']].apply(lambda x: x.describe())
print(agg_grouped)
print()

# 변환:z-score 계산
def z_score(x):
    return (x-x.mean()) / x.std()


age_zscore = grouped[['age','survived']].apply(z_score)
print(age_zscore)
print()

age_filter =grouped[['age','survived']].apply(lambda x:x['age'].mean()<30)
print(age_filter)
print()

print(df.loc[df['class'].isin(age_filter[age_filter==True].index),['age','survived']])
print()










