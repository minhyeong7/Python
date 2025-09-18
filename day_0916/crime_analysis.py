import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# pd.set_option('display.max_rows', None) 

pd.set_option('display.unicode.east_asian_width',True)

# 한글 폰트 설정 및 마이너스 기호 깨짐 방지
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 사용할 한글 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

file_path_loc = "data\\경찰청_범죄 발생 장소별 통계_20241231.csv"
file_path_reg="data\\경찰청_범죄 발생 지역별 통계_20231231.csv"

df_loc = pd.read_csv(file_path_loc, encoding="cp949")
df_reg = pd.read_csv(file_path_reg, encoding="cp949")


# 행과 열의 갯수
print("장소별 행과열:",df_loc.shape)
print()
print("지역별 행과열:",df_reg.shape)
print()

# 컬럼별 널값 개수 확인
print(df_loc.isnull().sum())
print()
print(df_reg.isnull().sum())
print()

# 숫자형 컬럼만 합계
loc_sum = df_loc.sum(numeric_only=True)
print(loc_sum)
print()

reg_sum = df_reg.sum(numeric_only=True)
print(reg_sum)
print()


# 장소별 합계 정렬
loc_sum_sorted = loc_sum.sort_values(ascending=False)
print("장소별 범죄 건수 TOP:")
print(loc_sum_sorted.head(10))
print()

plt.figure(figsize=(16,10))
loc_sum_sorted.head(10).plot(kind='bar', color='orange')
plt.title("장소별 범죄 건수 TOP 10", fontsize=16)
plt.xlabel("장소",fontsize=20)
plt.ylabel("범죄 건수", fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.figure(figsize=(16,10))
loc_sum_sorted.tail(10).plot(kind='bar', color='orange')
plt.title("장소별 범죄 건수 하위 10", fontsize=16)
plt.xlabel("장소",fontsize=20)
plt.ylabel("범죄 건수", fontsize=12)
plt.xticks(rotation=45, ha='right')

# 지역별 합계 정렬
reg_sum_sorted = reg_sum.sort_values(ascending=False)
print("지역별 범죄 건수 TOP:")
print(reg_sum_sorted.head(10))
print()

plt.figure(figsize=(16,10))
reg_sum_sorted.head(10).plot(kind='bar', color='skyblue')
plt.title("지역별 범죄 건수 TOP 10", fontsize=16)
plt.xlabel("지역",fontsize=12)
plt.ylabel("범죄 건수", fontsize=12)
plt.xticks(rotation=45, ha='right')



# 범죄대분류별 합계 계산 (숫자 컬럼만)
crime_sum_loc = df_loc.groupby('범죄대분류').sum(numeric_only=True)


# 전체 합계 기준 내림차순 정렬
crime_sum_sorted_loc = crime_sum_loc.sum(axis=1).sort_values(ascending=False)

print("범죄대분류별 총 범죄 건수 TOP:")
print(crime_sum_sorted_loc)
print()
import matplotlib.pyplot as plt

# 막대그래프 그리기
plt.figure(figsize=(16,10))
crime_sum_sorted_loc.plot(kind='bar', color='skyblue')

# 그래프 제목과 축 라벨
plt.title("범죄대분류별 총 범죄 건수", fontsize=16)
plt.ylabel("범죄 건수", fontsize=12)
plt.xticks(rotation=45, ha='right')  # x축 라벨 회전




# 범죄대분류별 합계 계산 (숫자 컬럼만)
crime_sum_reg = df_reg.groupby('범죄대분류').sum(numeric_only=True)

# 전체 합계 기준 내림차순 정렬
crime_sum_sorted_reg = crime_sum_reg.sum(axis=1).sort_values(ascending=False)

print("범죄대분류별 총 범죄 건수 TOP:")
print(crime_sum_sorted_reg)
print()

# 컬럼명에 '외국' 키워드가 들어간 컬럼만 선택
foreign_cols = [col for col in df_reg.columns if '외국' in col]

# 선택한 컬럼의 총 합계 계산
foreign_sum = df_reg[foreign_cols].sum(numeric_only=True)

print("외국 관련 컬럼 합계:")
print(foreign_sum)

# 막대그래프 그리기
plt.figure(figsize=(16,10))
foreign_sum.plot(kind='bar', color='skyblue')

# 그래프 제목과 축 라벨
plt.title("외국인 관련 범죄 합계", fontsize=16)
plt.xlabel("외국인 국적",fontsize=12)
plt.ylabel("범죄 건수", fontsize=12)
plt.xticks(rotation=45, ha='right')  # x축 라벨 회전


# '범죄대분류' 컬럼에서 '강력범죄'만 필터
violent_crime = df_loc[df_loc['범죄대분류'] == '강력범죄']
# 숫자형 컬럼만 선택해서 합계 계산 (장소별)
violent_loc_sum = violent_crime.sum(numeric_only=True)
violent_loc_sorted = violent_loc_sum.sort_values(ascending=False)

plt.figure(figsize=(12,8))
violent_loc_sorted.head(10).plot(kind='bar', color='red')
plt.title("강력범죄 장소별 발생 건수 TOP 10")
plt.ylabel("범죄 건수")
plt.xticks(rotation=45, ha='right')


# df_loc 음수 확인
neg_loc = (df_loc.select_dtypes(include=['int64','float64']) < 0).sum()
print(" df_loc 음수값 개수:")
print(neg_loc[neg_loc > 0])

# df_reg 음수 확인
neg_reg = (df_reg.select_dtypes(include=['int64','float64']) < 0).sum()
print("\n df_reg 음수값 개수:")
print(neg_reg[neg_reg > 0])



# plt.show()

