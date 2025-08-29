import re

print("-----DOTALL-----")
print()
p= re.compile('a.b')
m=p.match('a\nb') # 라인개행만큼은 못찾음
print(m)

p=re.compile('a.b',re.DOTALL)
m=p.match('a\nb') # 라인개행도 찾을 수 있음
print(m)

print()
print("-----IGNORECASE-----")

p= re.compile('[a-z]+',re.I) # 대소문자 무시하고 찾을 수 있음
m=p.match('python')
print(m)
m=p.match('Python')
print(m)
m=p.match('PYTHON')
print(m)

print()
print("-----MULTILINE-----")
p=re.compile("^python\s\w+",re.M) # 줄 시작 했을때 문장이 python이고 그다음 단어
data ='''python one life is too short 
python two you need python 
python three'''

m = p.findall(data)
print(m)

print("------문장의 첫 단어 추출 (멀티 라인)-------")

text = """
Hello world
안녕하세요 파이썬Reg
ex is pwoerful
"""
pattern=r"^\w+"

p=re.compile(pattern,re.M)
m=p.findall(text)
print(m)

