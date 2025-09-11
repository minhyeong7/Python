import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font',family=font_name)
plt.rcParams['axes.unicode_minus'] =False
pd.set_option('display.width',500)

samsung_revenue = pd.read_csv('./data/삼성전자_분기별_매출액.csv')
samsung_revenue.sort_values(by='quarter',inplace=True)
print(samsung_revenue)
print()

# fig = plt.figure() # 기본 640X480 픽셀 
# print(fig) 
# print()

# fig, ax = plt.subplots()
# print(fig)
# print(ax)
# plt.show()


# # ---- 그려보기----
# fig, ax = plt.subplots(figsize=(8,2))
# ax.plot(samsung_revenue['quarter'], samsung_revenue['value'])

# ax.annotate('테스트',
#             xy=(1,6.5e13), # '2023-Q3' 이렇게 표기도 가능

#             )
# plt.show()

# ------ axes 두개 불러와 보기 -------
# fig, axes= plt.subplots(2,1 , figsize=(12,5))

# print(fig)
# print(axes)

# plt.show()


# ----- 그려 보기 ------
fig, axes = plt.subplots(2,2,figsize=(12,2))
axes[0][0].plot(samsung_revenue['quarter'], samsung_revenue['value'])
samsung_revenue['value'].plot(ax=axes[1][0], marker='<')

print(fig)
print(axes)

plt.show()
