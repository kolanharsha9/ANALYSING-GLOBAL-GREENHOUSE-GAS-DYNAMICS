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

#%%

country_list = [
    'United States', 'China, P.R.: Mainland', 'Japan', 'Germany', 'India',
    'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada', 'South Korea',
    'Australia', 'Spain', 'Mexico', 'Indonesia', 'Netherlands, The',
    'Saudi Arabia', 'Turkey', 'Switzerland', 'Taiwan Province of China',
    'Sweden', 'Poland, Rep. of', 'Belgium', 'Thailand', 'Iran, Islamic Rep. of',
    'Austria', 'Norway', 'United Arab Emirates', 'Nigeria', 'Israel',
    'South Africa', 'Egypt, Arab Rep. of', 'Malaysia', 'Singapore',
    'Hong Kong SAR, China', 'Ireland', 'Denmark', 'Philippines', 'Finland',
    'Pakistan', 'Chile', 'Bangladesh', 'Vietnam', 'Greece', 'Colombia',
    'Ukraine', 'Romania', 'Czech Rep.', 'Portugal', 'Peru', 'Iraq', 'Qatar',
    'Algeria', 'Kuwait', 'Morocco', 'Hungary', 'Kazakhstan, Rep. of', 'Angola',
    'Sri Lanka', 'Ethiopia, The Federal Dem. Rep. of', 'Dominican Rep.', 'Kenya',
    'Oman', 'Venezuela, Rep. Bolivariana de', 'Luxembourg', 'Panama', 'Bulgaria',
    'Croatia, Rep. of', 'Myanmar', 'Sudan', 'Belarus, Rep. of', 'Costa Rica',
    'Uruguay', 'Tunisia', 'Uzbekistan, Rep. of', 'Slovakia, Rep. of',
    'Azerbaijan, Rep. of', 'Lebanon', 'Tanzania, United Rep. of', 'Ecuador',
    'Bolivia', 'Cameroon', 'Jordan', 'Bahrain, Kingdom of', 'Sri Lanka',
    'Bulgaria', 'Nepal', 'Iceland', 'Trinidad and Tobago', 'Estonia, Rep. of',
    'Slovenia, Rep. of', 'Paraguay', 'Cambodia', 'El Salvador', 'Latvia',
    'Papua New Guinea', 'Mozambique, Rep. of', 'Zambia', 'Cyprus', 'Gabon','Argentina','Russian Federation','China, P.R.: Macao'
]

df_filtered = data[data['Country'].isin(country_list)]

#%%
df_filtered.to_csv('C:/Users/YASH/Documents/Intro-to-Data-Mining-Project/data.csv', index=False)

#%%


# %%
