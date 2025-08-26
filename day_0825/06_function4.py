# 람다식
# 함수를 간단하게 만들기

def add(a, b):
    return a+b

add2 = lambda a, b: a+b

result = add2(3, 4)
print(result)
print()

distance = lambda x1, y1, x2, y2: ((x2-x1)**2 + (y2-y1)**2)**0.5

result = distance(1, 2, 4, 6) # 루트{(4-1)제곱 + (6-2)제곱}
print(result)
print()

print("-----리스트 + 맵-----")
print()

print("==============================")

numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))
print(squares) # [1, 4, 9, 16, 25]
print()

# 리스트컴프리헨션 방식으로 똑같은 작동 구현하기

squares2 = [x**2 for x in numbers]
print(squares2)

print("==============================")

numbers2 = [1, 2, 3, 4, 5, 6, 7, 8, 9,]
evens = list(filter(lambda x: x % 2 == 0, numbers2))
print(evens)  # [2, 4, 6, 8]
print()

# 리스트컴프리헨션 방식으로 똑같은 작동 구현하기

evens2 = [x for x in numbers2 if x % 2 == 0]
print(evens2)


print("==============================")

numbers3 = [5, -2, 0, 8, -7]
result = list(map(lambda x : "양수" if x > 0 else ("음수" if x < 0 else "0"), numbers3))
print(result)
print()

# 리스트컴프리헨션 방식으로 똑같은 작동 구현하기

result2 = ["양수" if x > 0 else ("음수" if x < 0 else "0") for x in numbers3]
print(result2)


print("==============================")

a = [1, 2, 3, 4]

result = [num * 3 for num in a if num % 2 == 0]
print(result) # [6, 12]
print()

# 람다 + 맵 + 필터 방식으로 똑같이 구현하기

result3 = list(map(lambda num: num * 3, filter(lambda x: x % 2 ==0, a)))
print(result3)
print()