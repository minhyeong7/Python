import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

print(titanic.head())
print()

sns.set_style('darkgrid')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.regplot(x='age', y='fare', data=titanic, ax=axes[0])
sns.regplot(x='age', y='fare', data=titanic, ax=axes[1], fit_reg=False)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(x='age', y='fare', hue='sex', data=titanic, ax=axes[0])
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic, ax=axes[1])
plt.show()