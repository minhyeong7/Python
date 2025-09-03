import pandas as pd
import seaborn as sns
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)


# 타이타닉 데이터 불러오기

titanic = sns.load_dataset('titanic') # 타이타닉 로드하기
titanic.info() # 데이터 구조확인
print()
print(titanic[['age','fare']].mean()) # 평균 나이 평균 요금
print()
# print(titanic['age',]) # age 결측치 age 평균으로 채우기
print()
ti=titanic.drop('deck',axis=1) # deck 컬럼제거
print()
print(titanic.loc[:,['age','parch','class']]) # age, parch,class 열만 보기
print()
mask=titanic['sibsp']+titanic['parch']+1
titanic['FamilySize']=mask # FamilySize컬럼 추가하기
print(titanic)
mask_age=titanic['age']<13 # IsChild 라는 컬럼 추가
titanic['IsChild']=mask_age 
print(titanic)
print()
# 남성여성 평균나이
mask_male=titanic['sex']=="male"
mask_female=titanic['sex']=="female"
print(titanic['age'])
print()
ndf=titanic.reset_index(drop=True)
nsdf = ndf.reset_index(names=['id']) # 기존 인덱스 지우고. + 숫자인덱스
print(ndf)
print()



# survived  생존여부 (1 / 0)
# pclass    선실 등급 (숫자형)
# sex       성별
# age       나이
# sibsp     함께 탑승한 형제/자매/배우자 수
# parch     함께 탑승함 부모/자녀 수
# fare      탑승요금
# embarked  탑승항구 (C = Cherbourg, Q = Queenstown, S = Southampton)
# class     선실 등급 (문자형)
# who       승객 구분 (man, woman, child)
# adult_male  성인 남성 (True, False)
# deck      선실 위치
# embark_town  탑승도시 이름 (Cherbourg, Queenstown, Southampton)
# alive     생존여부 (yes / no)
# alone     혼자야? (True / False)