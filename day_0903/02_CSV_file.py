import pandas as pd
import os 


# data 폴더 만들어서 read_csv_sample.csv 넣기

# 디렉토리 아래에 있는 폴더/파일 목록 반환
print(os.listdir('./data'))

file_path = "./data/read_csv_sample.csv"
print(file_path)

'''
os.path.join("폴더","파일명")
path2 = os.path.join("home","user","documents","file.txt")
'''

file_path2 = os.path.join('data','read_csv_sample.csv')
print(file_path2)
print()
# 현재 절대경로 반환
cur_dir = os.getcwd()
print(cur_dir)
print()
# 절대경로 지정
file_path3= os.path.join(cur_dir,"data","read_csv_sample.csv")
print(file_path3)
print()
# 마우스 우클릭으로 절대 경로 복사
file_path4="C:\\sers\\admin\Desktop\\PythonClass\\day_0903\\02_CSV_file.py"
print(file_path4)
print()

# 마우스 우클릭으로 상대경로 복사
file_path5="data\\read_csv_sample.csv"
print(file_path5)
print()

# 첫 행의 데이터가 열 이름이 된다.
# 자동 정수 인덱스
df1 = pd.read_csv(file_path)
print(df1)
print()


# 첫 행부터 데이터인 경우에는 None을 주자 
df2 = pd.read_csv(file_path,header=None)
print(df2)
print()


# 1번과 동일
df3 = pd.read_csv(file_path, index_col=None)
print(df3)
print(type(df3))
print()


df4 = pd.read_csv(file_path, index_col='c0')
print(df4)
print(type(df4))
print()
df4.info()


