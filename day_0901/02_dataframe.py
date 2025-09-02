# 데이터 프레임 == 이차원 자료구조 == 엑셀같은.
# 데이터프레임의 각 열은 각각 시리즈 객체이다.
# 행(row),  열(column),

# pandas 라이브러리 불러오기
import pandas as pd

# 출력 시 한글 등이 폭 맞게 표시되도록 설정
pd.set_option('display.unicode.east_asian_width', True)

print("----- 딕셔너리를 데이터프레임으로 -----\n")

# 딕셔너리 데이터 정의 (key=열 이름, value=리스트)
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9]}

# 딕셔너리를 DataFrame으로 변환
df = pd.DataFrame(dict_data)

print(type(df))  # 객체 타입 확인 (pandas DataFrame)
print(df)        # 데이터프레임 출력
print()

print("----- 이중 리스트를 데이터프레임으로 -----\n")

# 리스트 안에 리스트 구조를 DataFrame으로 변환 (행 단위)
df2 = pd.DataFrame([[15, '남', '덕영중'], [17, '여', '수리중']])
print(df2)
print()

# 행(index)과 열(columns) 이름 지정
df2 = pd.DataFrame(
    [[15, '남', '덕영중'], [17, '여', '수리중']],
    index=['준서', '예은'],                   # 행 이름
    columns=['나이', '성별', '학교']          # 열 이름
)
print(df2)
print()

# 행 인덱스와 열 이름 확인
print(df2.index)
print(df2.columns)
print()

# 인덱스와 컬럼 이름 변경
df2.index = ['학생1', '학생2']        # 행 이름 변경
df2.columns = ['연령', '남녀', '소속'] # 열 이름 변경
print(df2)
print()

print('-------- rename ---------\n')

# rename() 사용 예시
# columns={'연령':'age', '남녀':'gender', '소속':'school'} -> 열 이름 변경
# inplace=True -> 원본 DataFrame(df2)에 바로 적용
df2.rename(columns={'연령':'age', '남녀':'gender', '소속':'school'}, inplace=True)
print(df2)
print()

# index={'학생1':'김학생'} -> 행 이름 변경
df2.rename(index={'학생1':'김학생'}, inplace=True)
print(df2)
print()

print('-------- drop ---------\n')

# 예제용 데이터 생성 (각 과목별 점수)
exam_data = {'수학':[90, 80, 70], '영어':[88, 77, 66], '음악':[30, 40, 50]}

df =pd.DataFrame(exam_data,index=['철수','영희','미진'])
print(df)

print('---------drop / 행삭제 ------------')
print()

df1=df.drop('미진')
print(df1)

print()
df2=df.drop(['철수','미진'])
print(df2)

print()
df3=df.drop('철수',axis=0) #'철수'인 행 삭제
print(df3)
print()

df4=df.drop('철수',axis='index')
print(df4)
print()

df5=df.drop(index=['철수'])
print(df5)
print()


print('---------drop / 열삭제 ------------')
print()

df2=df.drop('수학',axis=1)
print(df2)
print()

df3= df.drop(['수학','영어'],axis=1) # 열 삭제 
print(df3)
print(df3.ndim)

print()

df4=df.drop('수학',axis='columns')
print(df4)
print()

df5=df.drop(columns=['수학'])
print(df5)
print()


