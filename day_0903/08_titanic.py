import pandas as pd
import seaborn as sns
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 300)


# 타이타닉 데이터 불러오기
titanic = sns.load_dataset('titanic') # 타이타닉 로드하기
titanic.info() # 데이터 구조확인
print()
print(titanic[['age','fare']].mean().round(2)) # 평균 나이 평균 요금
print()
age_mean=titanic['age'].mean().round(3)
titanic['age']=titanic['age'].fillna(age_mean)# age 결측치 age 평균으로 채우기
print(titanic)
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
# 불타입의 시리즈를 데이터 []에 넣으면 True에 해당하는 데이터만 필터링
# 남성과 여성의 평균 나이 비교

age_man=titanic[titanic['sex']=="male"]['age'].mean()
age_female=titanic[titanic['sex']=="female"]['age'].mean()
print(age_man)
print(age_female)
print()

titanic=titanic.reset_index(names='id') # id라는 인덱스 추가
titanic=titanic.set_index('id') # 기존 인덱스 삭제
print(titanic)


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



