import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({
    'color' : ['red','green','blue','red'],
    'size'  : ['S','M','L','S'],
    'shape' :   ['circle','square','triangle','cirlce']


})
print(data)
print()

encoder = OneHotEncoder(sparse_output=False)
print(data.ndim)
print()

encoded = encoder.fit_transform(data)

print(encoded)
print()

