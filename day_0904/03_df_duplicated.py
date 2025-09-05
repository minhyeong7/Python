import pandas as pd


df = pd.DataFrame({'c1':['a','a','b','a','b'],
                   'c2':[1,1,1,2,2],
                   'c3':[1,1,2,2,2]
                   })


print(df)
print()


print("--------- 중복데이터 확인 -----------")
print()

# 데이터프레임 전체 행 중에서 중복값 찾기
df_dup_first= df.duplicated() # 디폴트 옵션 keep='first'
print(df_dup_first)
print()

# 데이터프레임 전체 행중에서 중복값 찾기(keep='last')
df_dup_last=df.duplicated(keep='last')
print(df_dup_last)
print()

# 데이터프레임 전체 행 중에서 중복값 찾기(keep=False)
df_dup_false=df.duplicated(keep=False)
print(df_dup_false)
print()


col_dup =df['c2'].duplicated()
print(col_dup)
print()

col_dup2=df.duplicated(subset=['c2'])
print(col_dup2)
print()

col_dup3 =df.duplicated(subset=['c2','c3'])
print(col_dup3)
print()


print("---------- 중복데이터 제거 ----------")
print()

print(df)
print()

# 데이터프레임에서 중복 행을 제거(기본값, keep='first') 앞에 유지
df2 = df.drop_duplicates()
print(df2)
print()

# 뒤에 유지
df3 = df.drop_duplicates(keep='last')
print(df3)
print()

# 다제거
df4 = df.drop_duplicates(keep=False)
print(df4)
print()

# c2, c3 열을 기준으로 중복 행을 (첫 중복) 제거
df5 = df.drop_duplicates(subset=['c2','c3'])
print(df5)
print()


# c2, c3 열을 기준으로 중복 행을 모두 제거
df6 = df.drop_duplicates(subset=['c2','c3'],keep=False)
print(df6)








