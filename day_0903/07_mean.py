import pandas as pd


pd.set_option('display.unicode.east_asian_width',True)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)




df = pd.read_csv("data\\auto-mpg.csv",header = None)

# 열 이름 저장
df.columns = ['mpg','cylinders','displacement','horsepower',
              'weight','acceleration','model year','origin',
              'name']

print(df)
print()

print("-----평균값------")
print()
print(df.mean(numeric_only=True))
print()
print(df['mpg'].mean())
print()
print()
print(df[['mpg','weight']].mean())
print()

print("-----중앙값------")
print()
print(df.median(numeric_only=True))
print()
print(df[['mpg','weight']])
print()
print(df['mpg'].median())

print("-----최댓값------")
print()
print(df.max(numeric_only=True))
print()
print(df['mpg'].max())
print()

print("-----최솟값------")
print()
print(df.min(numeric_only=True))
print()
print(df['mpg'].min())
print()

print("-----표준편차------")
print()
print(df.std(numeric_only=True))
print()
print(df['mpg'].std())
print()

print("-----분산------")
print()
print(df.var(numeric_only=True))
print()
print(df['mpg'].var())
print()

print("-----상관계수------")
print()
print(df.corr(numeric_only=True))
print()
print(df[['mpg','weight']].corr())
print()


