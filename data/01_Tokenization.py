sample_text='''
    Hello! This is an example sentence for NLP preprocessing.
    Let's clean, tokenize, and get ready for modeling!
'''


import re

def clean_text(text):
    text = text.lower() # 소문자화
    text = re.sub(r'\d','',text) # 숫자 제거
    text = re.sub(r"[!,.]",'',text)# 특수문자 제거
    text= text.strip() # 양쪽 공백제거
    return text

cleaned =clean_text(sample_text)
print(cleaned)