# 리스트 컴프리헨션 
# list comprehension

a = [1, 2, 3, 4]
result = []
for num in a:
    result.append(num*3)

print(result)  # [3, 6, 9, 12]
print()

print("-----리스트 컴프리헨션-----")
print()

a = [1, 2, 3, 4]
result = [num * 3 for num in a]
print(result) # [3, 6, 9, 12]
print()

result = [num * 3 for num in a if num % 2 == 0]
print(result) # [6, 12]
print()

# [결과 for 항목 in 리스트(튜플) if 문 ]
# for문을 2개 이상 사용하는 것도 가능 하다!

result = [x*y for x in range(2,10)
              for y in range(1,10)]
print(result)
print()

print("-----break-----")
print()

for i in range(10):
    print(i, end=' ')
    if i == 5:
        break
print()

print("안녕히 주무세요")
print()

print("-----for-else 문-----")
print()

for i in range(5):
    print(i)
else:
    print("for문 정상종료.")

print()

for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("for문 정상종료???")

print()

print("-----enumerate 함수-----")
print()

fruits = ['apple', 'banana', 'orange']
for i, fruit in enumerate(fruits):
    print(f'{i}: {fruit}')

print()

for i, fruit in enumerate(fruits, 1):  # 1부터 시작
    print(f'{i}: {fruit}')

print()

print("-----zip 함수-----")
print()

names = ['홍길동', '김철수', '이영희']
scores = [85, 93, 56]

a = zip(names, scores)
print(a)
print(list(a))
print(list(a))
print()

# [('홍길동', 85), ('김철수', 93), ('이영희', 56)]
for name, score in zip(names, scores):
    print(f'{name}: {score}점')

print()


names = ['홍길동', '김철수', '이영희', '박영수']
scores = [85, 93, 56]

for name, score in zip(names, scores): # 개수가 안맞을 경우 무시됨.
    print(f'{name}: {score}점')

print()

# zip_longest 임포트!
from itertools import zip_longest

for name, score in zip_longest(names, scores, fillvalue="점수 없음"): 
    print(f'{name}: {score}')

print()

names = ['홍길동', '김철수', '이영희']
korean = [85, 93, 56]
english = [90, 100, 95]

# 홍길동 : 국어 85점, 영어 90점