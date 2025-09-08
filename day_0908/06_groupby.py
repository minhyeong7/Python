import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:,['age','sex','class','fare','survived']]

print(df.head())
print()

# class 열을 기준으로 그룹화 
grouped = df.groupby(['class'],observed=True)
print(grouped)
print()

for key,group in grouped:
    print('key :', key)
    print('number :', len(group))
    print(group.head())
    print()

print()


# 연산 메서드 적용
average = grouped.mean(numeric_only=True)
print(average)
print()

group3 = grouped.get_group(('Third',))
print(group3.head())
print()

group_two = df.groupby(['class','sex'],observed=True)


for key,group in group_two:
    print('key :', key)
    print('number :', len(group))
    print(group.head())
    print()

print()

# group_two 에 연산 메서드 적용
average_two = group_two.mean(numeric_only=True)
print(average_two)
print()

# 그룹바이로 'Third', 'female' 만 뽑기
group3f = group_two.get_group(('Third','female'))
print(group3f.head())
print()

# 필터링으로 'Third','female'만 뽑기

mask1 = df['class'] == "Third"
mask2 = df['sex'] == "female"
df_boolean = df[mask1 & mask2]
print(df_boolean)










