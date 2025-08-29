import re

s='<html><head><title>Title</title>'

print(len(s))

print(re.match('<.*>',s).span())
print(re.match('<.*>',s).group())

print(re.match('<.*?>',s).group())

text='aaaaaaaa'
pattern=r'a{2,4}?'

matches=re.findall(pattern,text)
print(matches)


print()
print("-----괄호 안의 내용 뽑기-----")
text = '오늘 메뉴는 (자장면) 과 (오징어덮밥) 입니다'

#['(자장면)','(오징어덮밥)']

pattern=r'\(.*\){5,7}?'

matches=re.findall(pattern,text)
print(matches)