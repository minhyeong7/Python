import pandas as pd

pd.set_option('display.unicode.east_asian_width',True)

exam_data = {'수학':[90, 80, 70], '영어':[88, 77, 66], '음악':[30, 40, 50]}

df = pd.DataFrame(exam_data, index=['철수','영희','미진'])
print(df)
print()

print("------ 행 선택------")
print()

label1=df.loc['철수']
label2=df.iloc[0]
label3=df.loc[['철수']]
df.iloc[0]

print(label1)
print()

print(label2)
print()

print(label3)
print()

print(type(label1))
print(type(label2))
print(type(label3))
print()

label4 = df.loc[['철수','영희']]
label5= df.iloc[[0,1]]

print(label4)
print()
print(label5)
print()

label6=df.loc['철수':'영희'] # 영희 포함
label7=df.iloc[0:1] # 1 미포함

print(label6)
print()
print(label7)
print()
print(type(label6))
print(type(label7)) # 한 줄이지만 범위로 뽑았으므로 데이터프레임

print("----- 열 선택-----")
print()

math1 = df['수학']
print(math1)
print(type(math1))
print()

eng = df.영어
print(eng)
print(type(eng))
print()

math_eng = df[['수학','영어']]
print(math_eng)
print(type(math_eng))
print()


math2 = df[['수학']] # 데이터 프레임 괄호 씌우면
print(math2)
print(type(math2))
print()

print("----- 고급 슬라이싱 -----\n")

# 행 선택: 0~2행까지, 2간격(step) → 0번째, 2번째 행
df3 = df.iloc[0:3:2]  
print(df3)

# 행 선택: 전체 행, 2간격 → 0, 2, 4...행
df3 = df.iloc[::2]    
print(df3)

# 행 선택: 전체 행을 역순으로
df3 = df.iloc[::-1]   
print(df3)

# 행 선택: 0~1행 (0,1) 슬라이싱
df3 = df.iloc[0:2]    
print(df3)
print()

# 행, 열 모두 선택: 0~1행, 0~1열
df3 = df.iloc[0:2, 0:2]  
print(df3)

# 모든 행, 0~1열 선택
df3 = df.iloc[:, 0:2]    
print(df3)

# 행 선택: 0~1행 (0,1)
df3 = df.iloc[0:2]        
print(df3)


