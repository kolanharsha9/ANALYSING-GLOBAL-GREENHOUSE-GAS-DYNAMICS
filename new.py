#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset =pd.read_csv("data.csv")


# %%
cols = ['Indicator','Unit','Source']
dataset = dataset.drop(cols, axis=1)
# %%
dataset.columns
# %%
dataset.to_csv("data.csv",index=False)

#%%
print(dataset.columns)

cols1 = ['F1970', 'F1971', 'F1972', 'F1973', 'F1974', 'F1975', 'F1976',
       'F1977', 'F1978', 'F1979', 'F1980', 'F1981', 'F1982', 'F1983', 'F1984',
       'F1985']

dataset = dataset.drop(cols1, axis = 1)

#%%
print(dataset.columns)
#%%
dataset.to_csv("data.csv",index = False)
# %%


# Load the dataset
# Assuming df is your DataFrame
# Replace 'your_dataset.csv' with the actual file name or provide the DataFrame directly
# df = pd.read_csv('your_dataset.csv')



# Load the dataset
# Assuming df is your DataFrame
# Replace 'your_dataset.csv' with the actual file name or provide the DataFrame directly
# df = pd.read_csv('your_dataset.csv')

# Filter rows where 'Industry' is not 'Not Applicable'
filtered_df = dataset[dataset['Industry'] != 'Not Applicable']

# Create a dictionary to map each industry to its corresponding category
industry_categories = {
    'Agriculture': ['Agriculture'],
    'Buildings and Infrastructure': ['Buildings and other Sectors', 'Domestic Aviation', 'Domestic Navigation', 'Road Transportation', 'Railways', 'Other Transportation'],
    'Chemical and Manufacturing Industries': ['Chemical Industry', 'Industrial Processes and Product Use', 'Manufacturing Industries and Construction', 'Mineral Industry', 'Non-energy Products from Fuels and Solvent Use', 'Electronics Industry'],
    'Energy': ['Energy', 'Energy Industries', 'Fugitive Emissions from Fuels'],
    'Waste': ['Waste'],
    'Other': ['Other', 'Other Product Manufacture and Use', 'Other (Not specified elsewhere)'],
    'Metal Industry': ['Metal Industry']
}

# Convert values in emission columns to numeric
emission_columns = ['F2011', 'F2012', 'F2013', 'F2014', 'F2015', 'F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']
filtered_df[emission_columns] = filtered_df[emission_columns].apply(pd.to_numeric, errors='coerce')

# Create a new column 'Total GHG Emissions' based on the sum of emission columns
filtered_df['Total GHG Emissions'] = filtered_df[emission_columns].sum(axis=1)

# Create a new column 'Industry Category' based on the mapping
filtered_df['Industry_Category'] = filtered_df['Industry'].apply(lambda x: next((category for category, industries in industry_categories.items() if x in industries), 'Unknown'))

#colors = sns.color_palette('rocket', len(filtered_df))
custom_palette = sns.color_palette("rocket", n_colors=len(filtered_df['Industry_Category'].unique()))


# Plot the bar graph with the new grouping
plt.figure(figsize=(12, 8))
filtered_df.groupby('Industry_Category')['Total GHG Emissions'].sum().sort_values(ascending=False).plot(kind='bar', color= custom_palette)
plt.title('Total Greenhouse Gas Emissions by Industry Category')
plt.xlabel('Industry Category')
plt.ylabel('Total GHG Emissions (Million Metric tons)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.9)
plt.show()

#%%
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#%%

filtered_df = dataset[dataset['Industry'] != 'Not Applicable']

# List of top 20 countries
top_20_countries = [
    'United States', 'China, P.R.: Mainland', 'Japan', 'Germany', 'India',
    'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada', 'South Korea',
    'Australia', 'Spain', 'Mexico', 'Indonesia', 'Netherlands, The',
    'Saudi Arabia', 'Turkey', 'Switzerland', 'Taiwan Province of China'
]

# Filter data for top 20 countries
filtered_df_top_20 = filtered_df[filtered_df['Country'].isin(top_20_countries)]

# Convert values in emission columns to numeric
emission_columns = ['F2011', 'F2012', 'F2013', 'F2014', 'F2015', 'F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']
filtered_df_top_20[emission_columns] = filtered_df_top_20[emission_columns].apply(pd.to_numeric, errors='coerce')

# Create a new column 'Total GHG Emissions' based on the sum of emission columns
filtered_df_top_20['Total GHG Emissions'] = filtered_df_top_20[emission_columns].sum(axis=1)

# Create a dictionary to map each industry to its corresponding color
industry_colors = {
    'Agriculture': 'blue',
    'Buildings and Infrastructure': 'orange',
    'Chemical and Manufacturing Industries': 'green',
    'Energy': 'red',
    'Waste': 'purple',
    'Other': 'brown',
    'Metal Industry': 'pink'
}

# Create a new column 'Industry Category' based on the mapping
filtered_df_top_20['Industry Category'] = filtered_df_top_20['Industry'].apply(lambda x: next((category for category, industries in industry_categories.items() if x in industries), 'Unknown'))

# Create a new column 'Industry Color' based on the mapping
filtered_df_top_20['Industry Color'] = filtered_df_top_20['Industry Category'].map(industry_colors)

# Plot the line chart with different lines for different industries
plt.figure(figsize=(12, 8))
for category, data in filtered_df_top_20.groupby('Industry Category'):
    plt.scatter(data['Country'], data['Total GHG Emissions'], label=category, color=data['Industry Color'].iloc[0], marker='o')

plt.title('Greenhouse Gas Emissions by Industry Category (Top 20 Countries)')
plt.xlabel('Country')
plt.ylabel('Total GHG Emissions')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
#%%

print("---------------------------------------------------------------------------------------------------------------------------------------------------")

# %%
filtered_df = dataset[dataset['Industry'] != 'Not Applicable']

# List of top 20 countries
bottom_20_countries = [
    'El Salvador', 'Cambodia', 'Paraguay', 'Slovenia, Rep. of', 'Estonia, Rep. of',
    'Trinidad and Tobago', 'Iceland', 'Nepal', 'Bulgaria', 'Sri Lanka',
    'Bahrain, Kingdom of', 'Jordan', 'Cameroon', 'Bolivia', 'Ecuador',
    'Tanzania, United Rep. of', 'Lebanon', 'Azerbaijan, Rep. of', 'Slovakia, Rep. of',
    'Uzbekistan, Rep. of'
]

# Filter data for top 20 countries
filtered_df_bottom_20 = filtered_df[filtered_df['Country'].isin(bottom_20_countries)]

# Convert values in emission columns to numeric
emission_columns = ['F2011', 'F2012', 'F2013', 'F2014', 'F2015', 'F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']
filtered_df_bottom_20[emission_columns] = filtered_df_bottom_20[emission_columns].apply(pd.to_numeric, errors='coerce')

# Create a new column 'Total GHG Emissions' based on the sum of emission columns
filtered_df_bottom_20['Total GHG Emissions'] = filtered_df_bottom_20[emission_columns].sum(axis=1)

# Create a dictionary to map each industry to its corresponding color
industry_colors = {
    'Agriculture': 'blue',
    'Buildings and Infrastructure': 'orange',
    'Chemical and Manufacturing Industries': 'green',
    'Energy': 'red',
    'Waste': 'purple',
    'Other': 'brown',
    'Metal Industry': 'pink'
}

# Create a new column 'Industry Category' based on the mapping
filtered_df_bottom_20['Industry Category'] = filtered_df_bottom_20['Industry'].apply(lambda x: next((category for category, industries in industry_categories.items() if x in industries), 'Unknown'))

# Create a new column 'Industry Color' based on the mapping
filtered_df_bottom_20['Industry Color'] = filtered_df_bottom_20['Industry Category'].map(industry_colors)

# Plot the line chart with different lines for different industries
plt.figure(figsize=(12, 8))
for category, data in filtered_df_bottom_20.groupby('Industry Category'):
    plt.scatter(data['Country'], data['Total GHG Emissions'], label=category, color=data['Industry Color'].iloc[0],marker='o')

plt.title('Greenhouse Gas Emissions by Industry Category (Bottom 20 Countries)')
plt.xlabel('Country')
plt.ylabel('Total GHG Emissions')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# %%
