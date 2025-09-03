import pandas as pd
import seaborn as sns
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)

titanic = sns.load_dataset('titanic')
print(titanic)
print(titanic.head(10))
titanic.info()

df = titanic.loc[0:9,['age','fare']]
print(df)
print()
print(df['age'])
print()
print(df['age']<20)
print()
print(df[df['age'] < 20]) # 중요하다 데이터프레임으로 만들기
print()
print(df.loc[df['age']<20]) #위에꺼랑 같음
print("-----논리연산자-----")
print(df.loc[~(df['age']<20)])
print()

# ~ &
#   10살 이상 & 20살 미만
print(titanic[(titanic['age']>=10) & (titanic['age']<20)])

mask1=(titanic['age']>=10) & (titanic['age']<20)
print(type(mask1))

df_teenage = titanic[mask1]
print(df_teenage.head())

mask2=(titanic['age']<10) & (titanic['sex']=='female')
df_female_under = titanic[mask2]
print(df_female_under.head())
print()

print("---------- 행 컨디션 열 셀렉션 -----------")
df_female_under10=titanic.loc[mask2,['age','sex']]
print(df_female_under10.head())
print()

mask3=(titanic['age']<10) | (titanic['age'] >60)
df_young_old=titanic.loc[mask3,['age','who']]
print(df_young_old.iloc[10:20])
print()

print(mask3)

titanic['df_y_o']=mask3
print(titanic)

print()

mask1 = titanic['embark_town'] == "Southampton"
mask2 = titanic['embark_town'] == "Queenstown"
df_boolean = titanic[mask1 | mask2]
print(df_boolean)


