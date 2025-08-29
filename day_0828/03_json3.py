# 리스트를 제이슨으로 저장/로드 하기

import json
import os

filename = "people2.json"

if not os.path.exists(filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump([], f)


# 제이슨 파일 불러오기
with open(filename, "r", encoding="utf-8") as f:
    loaded_people = json.load(f)


name = input("이름 입력: ")
age = int(input("나이 입력: "))

loaded_people.append({"name": name, "age":age})


# 제이슨으로 덤프하기
with open(filename, "w", encoding="utf-8") as f:
    json.dump(loaded_people, f, ensure_ascii=False, indent=4)