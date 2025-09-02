import pandas as pd 
pd.set_option('display.unicode.east_asian_width',True)


students={
        '이름':['서준','우현','인아'],
        '수학':[90,80,70],
        '영어':[50,40,70],
        '사회':[30,10,20]
}

df = pd.DataFrame(students)
print(df)
print()

df.index=['가','나','다']
print(df)
print()


df=df.set_index('이름')
print(df)
print()

print("------ 열 추가하기------")
print()

df['국어']=80
print(df)
print(df.shape) # 행열 갯수 파악


print("----- 태뷸릿 적용해보기 -----")
print()
from tabulate import tabulate

print(tabulate(df,headers='keys',tablefmt='psql'))

df['미술']= [89,90,91]
df['국어']=[70,53,32]
print(df)

df.loc['동수']=0
print(df)
df.loc['말숙'] = [34,54,77,88,23]

print("------원소 값 변경------")
print()

df.loc['동수','수학']=60
print(df)

df.iloc[3,1]=70
print(df)

df.loc['동수','사회':'미술']=90,80,100
print(df)

print("----- 행, 열 자리 바꾸기 (전치) -----")
print()

df = df.transpose()
print(df)

df = df.T
print(df)





