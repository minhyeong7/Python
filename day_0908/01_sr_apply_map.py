# 시리즈 원소에 함수 매핑

import seaborn as sns
import pandas as pd


# titanic 데이터셋에서 age, fare 열 뽑아서 df 만들기
titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age','fare']]
print(df.head())
print()

# 사용자 함수 정의
def add_10(n):
    return n+10

def add_two_obj(a,b):
    return a+b

print(add_10(10))
print()
print(add_two_obj(10,10))
print()

# 시리즈 객체에 apply 함수 적용
sr1 = df['age'].apply(add_10)
print(sr1)
print()

sr2 = df['age'].apply(lambda n: n+10)
print(sr2)
print()


# # 함수의 매개변수가 2개 이상인 경우
# sr3 = df['age'].apply(add_two_obj, b=10)
# print(sr3.head())

# # 함수의 매개변수가 2개 이상인 경우 (람다 함수로!)
# sr4 = df['age'].apply(lambda a,b : a+b, b=10)
# print(sr4)
# print()

# # 시리즈의 원소에 map() 적용
# def over_thirty(age):
#     return age>30 

# sr_map = df['age'].map(over_thirty)
# print(sr_map.head())
# print()

# # apply와 동일함
# # sr_map2 = df['age'].apply(over_thirty)
# # print(sr_map2.head())
# # print()

# sr_map2=df['age'].map(lambda age: True if age>30 else False)
# print(sr_map2.head())
# print()

# print(titanic['sex'].unique())
# print()
# print(titanic['sex'].head())
# print()

# gender_dict={'male':0,'female':1}
# titanic['gender']=titanic['sex'].map(gender_dict)
# titanic.info()
# print(titanic)




