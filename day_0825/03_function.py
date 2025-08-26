# 함수

def add(a, b):
    return a + b

hap = add(3, 4) # 숫자로 넣어서, 변수로 받아서 출력
print(hap)

c = 5
d = 6

hap = add(c, d) # 변수로 넣어서, 변수로 받아서 출력하기
print(hap)

print(add(8, 9))  # 숫자 넣어서, 바로 출력하기
print()

print("-----입력값, 반환값 있는 경우 -----")
print()

# 조금 더 정석적으로 표현
def add(a, b):      # a, b 는 매개변수(parameter)
    result = a + b
    return result

hap = add(4, 7)    # 4, 7 은 인수(arguments)
print(hap)
print()

print("-----입력값이 없는 함수-----")
print()

def say():
    return 'hi'

say()  # 아무것도 안나옴
a = say()
print(a) # hi
 
print("-----입력값, 반환값 없는 함수-----")

def say2():
    print("hi")

say2() # hi
a = say2() # hi
print(a) # None
print(say2()) # hi   # None

print("-----입력값만 있는 경우-----")
print()

def add2(a, b):
    print(f'{a}와 {b}의 합은 {a+b} 입니다.')

add2(3,4)


print("-----헷갈리게 혼합하기-----")
print()

def add2(a, b):
    print(f'{a}와 {b}의 합은 {a+b} 입니다.')
    print(f'하지만 반환하는 값은 {a} x {b} 입니다.')
    result = a * b
    return result

a = add2(3,4)
print(a)
print()

print("-----매개변수 지정하여 입력-----")
print()

def sub(a, b):
    return a - b

result = sub(a=3, b=4)
print(result)

result = sub(b=5, a=9)  # 매개변수 지정하면 순서 달라도 된다.
print(result)

# 4가지 경우 만들어 보세요~!
# 입력값 o x 반환값 o x