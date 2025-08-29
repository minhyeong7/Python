# !!! 그루핑 !!!
# 1. 여러문자를 하나로 묶어서 반복 처리
# 2. 매치된 문자열에서 원하는 부분만 추출
# ()
import re
p= re.compile('(ABC)+')
m=p.search('ABCABCABC OK?')
print(m) #<re.Match object; span=(0, 9), match='ABCABCABC'> 객체
print(m.group()) #ABCABCABC 우리가 원하는 문자 볼 수 있음


p= re.compile(r"\w+\s+\d+[-]\d+[-]\d+")
m=p.search("park 010-1234-5678")

# 이름 부분만 추출
p= re.compile(r"(\w+)\s+((\d+)[-]\d+[-]\d+)")
m=p.search("park 010-1234-5678")
print(m)
print(m.group(1))
print(m.group(2))
print(m.group(3))

# 문자열 재참조
p = re.compile(r'(\b\w+)\s+\1')
m=p.search('Paris in the the spring').group()
print(m)


print("-----이메일 사용자명과 도메인 분리-----")
text='문의: hello.world@python.org'
pattern=r"(([A-Za-z.]+)@([A-Za-z]+\.[A-Za-z]{2,}))"

match=re.search(pattern,text)
print("전체:",match.group(1))
print("사용자명:",match.group(2))
print("도메인명:",match.group(3))



print()
print("-----중복된 글자 줄이기------")
# 3개이상 된거를 줄임
text = "굿굿굿 와아아!!! 대박"
pattern = r"(.)\1{2,}"

result = re.sub(pattern,r"\1\1",text)
print("중복 줄이기:",result)

print("---전화번호 정규화(하이픈 통일)-----")
# 0으로 시작하여 
# 맨앞이 0포함 2~3자리
# 가운데가 3~4자리
# 끝이 4자리
text="고객센터 02-123-1234,01012341234,031.123.1234 (대표)"
rx=re.compile(r"(0[0-9]{1,2})[-. ]?([0-9]{3,4})[-. ]?([0-9]{4})")

normalized=rx.sub(r"\1-\2-\3",text)
print("정규화:",normalized)


# finditer 로 각 그룹 확인
m = rx.finditer(text)
print(m)

for i in m:
    print("원본:",i.group(0),"지역번호:",i.group(1),"국번호:",i.group(2),"가입자:",i.group(3))


