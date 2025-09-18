import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기 (헤더 없음)
df = pd.read_csv('./data/auto-mpg.csv', header=None)

# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

plt.style.use('default')  # 기본 스타일 사용

print(df)  # 전체 데이터 출력
print()

# 새로운 열 'count' 생성, 각 행 1로 설정 (갯수 카운트용)
df['count'] = 1
print(df)  # 'count' 열 추가 확인
print()

# origin(제조국가)별 합계 계산 (숫자 컬럼만)
df_origin = df.groupby('origin').sum(numeric_only=True)
print(df_origin)  # 그룹 합계 출력
print()

# index를 숫자 → 국가명으로 변경
df_origin.index = ['USA', 'EU', 'JAPAN']
print(df_origin)  # 변경 확인
print()

# 문자열 포매팅 예제
print('나는 %d 살 입니다.' % 23)

# ---------------- 파이 차트 그리기 ----------------
df_origin['count'].plot(
    kind='pie',               # 파이 차트
    figsize=(7,5),            # 그림 크기
    autopct='%1.1f%%',        # 퍼센트 표시 (소수점 1자리)
    startangle=10,            # 시작 각도
    colors=['chocolate','bisque', 'cadetblue']  # 파이 색상 지정
)

plt.title('Model Origin', size=20)  # 제목 설정
plt.axis('equal')                   # 원형 비율 유지
plt.legend(labels=df_origin.index)  # 범례 표시
plt.show()
