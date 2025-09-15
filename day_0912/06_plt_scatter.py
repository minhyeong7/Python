import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')

df = pd.read_csv('./data/auto-mpg.csv', header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']
print(df)
print()

# 연비(mpg)와 차중(weight) 열에 대한 산점도 그리기
df.plot(kind='scatter', x='weight', y='mpg', c='coral', s=10, marker='d', figsize=(10,5))
plt.title('Scatter Plot - mpg vs. wieght')
# plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df['weight'], df['mpg'], c='green', s=10)
plt.title('Scatter Plot - mpg vs. wieght')
plt.xlabel('weight')
plt.ylabel('mpg')
# plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='weight', y='mpg', color='coral', s=20)
plt.title('Scatter Plot - mpg vs. wieght')
plt.show()

# ------------ 버블 차트 -----------

# cylinders 개수의 상대적 비율을 계산하여 시리즈 생성
print(df['cylinders'].unique())
cylinders_size = (df['cylinders'] / df['cylinders'].max()) * 300
print(cylinders_size)

df.plot(kind='scatter', x='weight', y='mpg', c='coral', figsize=(10,5),
        s=cylinders_size, alpha=0.3)
# s 기본값 = 20(포인트) (1포인트 = 1/72인치)
plt.title('Scatter Plot - mpg vs. wieght - cylinders')
plt.show()

# ------------ 저장하기 ------------

df.plot(kind='scatter', x='weight', y='mpg', marker='+',
        cmap='rainbow', c=cylinders_size, s=50, figsize=(10,5), alpha=0.3)
# viridis, plasma, coolwarm, jet, rainbow
plt.title('Scatter Plot - mpg vs. wieght - cylinders')
plt.savefig('./data/scatter.png')
plt.savefig('./data/scatter_transparent.png', transparent=True)
plt.show()
