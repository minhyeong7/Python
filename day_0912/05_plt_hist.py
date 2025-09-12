import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('classic')

df = pd.read_csv('./data/auto-mpg.csv', header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df['mpg'].plot(kind='hist', bins=10, color='coral', figsize=(8,5))
plt.title('Histogram')
plt.xlabel('mpg')
# plt.show()

df[['mpg', 'origin']].plot(by=['origin'], kind='hist', figsize=(8,10))
# plt.show()

df[['mpg', 'origin', 'cylinders']].plot(by=['origin', 'cylinders'],
                                        kind='hist', figsize=(8, 30))
plt.show()