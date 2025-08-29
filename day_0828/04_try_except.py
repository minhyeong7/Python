#예외처리

# #FileNotFoundError
# f=open("나는 없는 파일","r")

# #ZeroDivisionError: division by zero
# a=4/0
# print(4/0)

# #IndexError: list index out of range
# a=[1,2,3]
# print(a[3])

print("---다양한 방법---")
try:
    a=4
    b=0
    c=4/0
    print(c)
except:
    print("0으로 나눌 수 없습니다")

print()

try:
    a=4
    b=0
    c=4/0
    print(c)
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다")

print()

try:
    a=4
    b=0
    c=4/0
    print(c)
except ZeroDivisionError as e:
    print(e)
print()

# # 여러가지 예외 처리하기
# print()
# print("-----여러가지 예외처리하기----")

# try:
#     x=int(input("분자 입력:"))
#     y=int(input("분모 입력:"))
#     result = x/y
#     print(f'결과:{result}')
# except ValueError:
#     print("숫자만 입력하세요")
# except ZeroDivisionError:
#     print("분모에 0을 넣을 수 없습니다")

# try:
#     x=int(input("분자 입력:"))
#     y=int(input("분모 입력:"))
#     result = x/y
#     print(f'결과:{result}')

# except Exception as e:
#     print("오류 발생:",e)

# else:
#     print("정상적으로 수행시 수행되는 문장입니다")

# finally:
#     print("무조건 실행되는 문장입니다")

