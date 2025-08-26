students = ['민수', '지영', '철수', '영희']
points = [3, -2, 0, 5]

print()
for name, point in zip(students, points):
    if point >= 5:
        status = '칭찬 대상'
    elif point > 0:
        status = '일반 학생'
    else:
        status = '주의 필요'
    
    print(f"{name} : {status}")


'''
5점 이상 : '칭찬 대상'
0점 초과 5점 미만 : '일반 학생'
0점 이하 : '주의 필요'
'''

# 최종 출력 결과 
'''
민수 : 일반 학생
지영 : 주의 필요
철수 : 주의 필요
영희 : 칭찬 대상
'''

# for 문, if 문, zip 함수 사용
# 변수 잘 활용해서 최대한 깔끔하게 코딩

