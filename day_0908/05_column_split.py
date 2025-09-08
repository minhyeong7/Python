import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)  # 한글·중국어·일본어 같은 동아시아 문자가 표에서 칸 맞춰 정렬되도록 설정

# 엑셀 데이터셋 불러오기

df = pd.read_excel('data\\주가데이터.xlsx')

print(df)

df.info()
print()

# 연, 월, 일 데이터 분리하기
df['연월일'] = df['연월일'].astype('str')
dates = df['연월일'].str.split('-')
print(dates)

# 분리된 정보를 각각 새로운 열에 담아서 df에 추가하기
df['열'] = dates.str.get(0)
df['월'] = dates.str.get(1)
df['일'] = dates.str.get(2)


print(df.head())

# expand 옵션
df_expand= df['연월일'].str.split("-",expand=True)
df_expand.columns = ['연','월','일']
print(df_expand.head())
print()

# -----------타임 스탬프 방법으로 하기-----------
df.info()
print()

df['연월일']=pd.to_datetime(df['연월일'])
df.info()
print()

# 연,월,일 바로 추출
df['연'] = df['연월일'].dt.year
df['월'] = df['연월일'].dt.month
df['일'] = df['연월일'].dt.day
df['요일'] = df['연월일'].dt.day_name()

print(df.head())



