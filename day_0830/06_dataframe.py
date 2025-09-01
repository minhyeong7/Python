import pandas as pd 
pd.set_option('display.unicode.east_asian_width',True)

students = {'이름':['서준','우현','인아'],
            '수학':[90,80,70],
            '영어':[55,66,77],
            '사회':[56,67,89],
            '체육':[10,20,30]
            }

df=pd.DataFrame(students)
print(df)
print()

print("-----인덱스 지정-----")
print()

ndf=df.set_index('이름')
print(ndf)
print()

ndf2= df.set_index(['이름'])
print(ndf2)
print()

ndf3=ndf2.set_index('수학')
print(ndf3)
print()

ndf33=ndf3.set_index('영어')
print(ndf33)
print()

ndf4=ndf2.set_index(['수학','영어'])
print(ndf4)
print()

print("-----인덱스 재배열-----")
print()

df = pd.DataFrame(students,index=['s1','s2','s3'])
print(df)
print()


new_index=['s1','s2','s3','s4','s5']

ndf=df.reindex(new_index)
print(ndf)
print()

ndf.loc[['s4','s5'],'수학']=[80,90]
print(ndf)
print()



ndf = df.reindex(new_index,fill_value=80)
print(ndf)

print("----- 인덱스/컬럼 재배열-----")
print()

ndf =df.reindex(index=new_index,columns=['이름','수학','영어','과학'])
print(ndf)

print()
print(df)
print()