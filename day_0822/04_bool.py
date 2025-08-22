# bool 자료형
# True: 참
# False: 거짓

a = True
b = False

print(type(a)) # <class 'bool'>
print(type(b)) # <class 'bool'>

print(1==1) # True
print(2 > 1) # True
print(2 < 1) # False

# 자료형에도 참과 거짓이 있다!?
# "hello" 참     "" 거짓
# [1,2,3] 참     [] 거짓
# (1,2,3) 참     () 거짓
# {'a':1} 참     {} 거짓
# 1       참     0  거짓   --- 0 이외의 숫자 = 참
# None 거짓

a = [1, 2, 3, 4]
while a:
    print(a.pop())

# while 조건문:
#      수행할_문장   
print()
print("----------")
if [1, 2, 3]: 
    print("참")
else:
    print("거짓")

print()
print("----------")
if []:
    print("참")
else:
    print("거짓")

print()
print("----------")
print(bool('python')) # True
print(bool('')) # False



print(bool([1]))
print(bool([]))

print(bool((1)))
print(bool(()))

print(bool(1))
print(bool(0))


