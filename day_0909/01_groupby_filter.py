import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:,['age','sex','class','fare','survived']]

# class 열을 기준으로 분할
grouped = df.groupby(['class'],observed=True)
print(grouped)
print()

# 그룹별로 첫 2행을 확인 
grouped_head = grouped.head(2)
print(grouped_head)
print()

# 각 그룹의 1번 인덱스 데이터를 확인
grouped_first= grouped.nth(1)
print(grouped_first)
print()

# 필터링이 적용된 그룹 객체의 열 부분 집합
print(grouped[['sex','survived']].nth(1))
print()

# 데이터 개수가 200개 이상인 그룹만을 필터링하여 반환
grouped_filter = grouped.filter(lambda x: len(x) >= 200 )
print(grouped_filter)
print()

# 각 그룹을 하나씩 꺼내 확인하는 방식
for key,group in grouped:
    print('key :', key)
    print('number :', len(group))
    print(group.head())
    print()

print()


# age 열의 평균이 30보다 작은 그룹만을 필터링하여 반환
age_filter = grouped.filter(lambda x: x['age'].mean() <30 )
print(age_filter)













