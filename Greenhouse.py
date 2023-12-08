#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("dataset.csv")
#%%
data.columns
#%%
data.info
#%%
percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_df = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': percent_missing})
#%%
missing_value_df.columns
#%%


plt.figure(figsize=(12, 5))

plt.xticks(fontsize=6)

plt.xticks(rotation=90)
sns.barplot(x='column_name',y='percent_missing',data=missing_value_df)
plt.show()
#%%
cols = ['F2022', 'F2023', 'F2024', 'F2025', 'F2026', 'F2027', 'F2028', 'F2029', 'F2030']
data = data.drop(cols, axis=1)
#%%
data.columns
#%%
data=data.dropna()
data.info()
# %%
data.columns
#%%
cols = ['ISO2', 'CTS_Full_Descriptor', 'CTS_Code', 'F1986', 'F1987', 'F1988', 'F1989', 'F1990', 'F1991', 'F1992', 'F1993', 'F1994', 'F1995', 'F1996', 'F1997', 'F1998', 'F1999', 'F2000']
data = data.drop(cols, axis=1)

