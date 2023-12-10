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
data1=df_filtered
data2 = data1[data1['Industry'] != 'Not Applicable']
# %%
# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

industry_categories = {
    'Agriculture': ['Agriculture'],
    'Buildings and Infrastructure': ['Buildings and other Sectors', 'Domestic Aviation', 'Domestic Navigation', 'Road Transportation', 'Railways', 'Other Transportation'],
    'Chemical and Manufacturing Industries': ['Chemical Industry', 'Industrial Processes and Product Use', 'Manufacturing Industries and Construction', 'Mineral Industry', 'Non-energy Products from Fuels and Solvent Use', 'Electronics Industry'],
    'Energy': ['Energy', 'Energy Industries', 'Fugitive Emissions from Fuels'],
    'Waste': ['Waste'],
    'Other': ['Other', 'Other Product Manufacture and Use', 'Other (Not specified elsewhere)'],
    'Metal Industry': ['Metal Industry']
}

# Function to map industries to categories
def map_industry_to_category(industry):
    for category, industries in industry_categories.items():
        if industry in industries:
            return category
    return 'Uncategorized'

# Add a new column 'Category' to the DataFrame
data2['Industry_Category'] = data2['Industry'].apply(map_industry_to_category)

# Print the updated DataFrame with the new 'Category' column
print(data2)
#%%
data2.to_csv('data.csv', index=False)

#%%
df=pd.read_csv('data.csv')

# %%
len(df['Country'].unique())
# %%
import matplotlib.pyplot as plt
import seaborn as sns



# List of columns representing years
year_columns = [f'F{year}' for year in range(2016, 2022)]

# Summing up GHG emissions by industry over the years
total_emissions_by_industry = df.groupby('Industry_Category')[year_columns].sum().sum(axis=1)

# Sorting industries by their total emissions
sorted_emissions = total_emissions_by_industry.sort_values(ascending=False)

# Plotting the results
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_emissions.values, y=sorted_emissions.index)
plt.title('Total GHG Emissions by Industry (2016-2021)')
plt.xlabel('Total Emissions (Million Metric Tons of CO2 Equivalent)')
plt.ylabel('Industry')
plt.show()






# %%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# List of top ten economy countries
top_ten_economy = [
    'United States', 'China, P.R.: Mainland', 'Japan', 'Germany', 'India',
    'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada'
]

# Filter the DataFrame to include only the top ten economy countries
df_top_ten = df[df['Country'].isin(top_ten_economy)]

# Select relevant columns for analysis (e.g., 'Country', 'F2016' to 'F2021')
columns_of_interest = ['Country', 'F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']
df_selected = df_top_ten[columns_of_interest]

# Group by 'Country' and calculate the mean emissions for each year
df_mean = df_selected.groupby('Country').mean()

# Plot the mean GHG emissions for each country from 2016 to 2021
plt.figure(figsize=(12, 8))
df_mean.T.plot(kind='line', marker='o')
plt.title('Mean GHG Emissions from 2016 to 2021 - Top Ten Economy Countries')
plt.xlabel('Year')
plt.ylabel('Mean GHG Emissions (Million metric tons of CO2 equivalent)')
plt.legend(title='Country', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()




# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with the specified columns

# Filter the DataFrame to include only rows with emissions data
emissions_df = df[df.iloc[:, 11:].notnull().any(axis=1)]

# Count occurrences of each (Industry, Gas_Type) combination
industry_gas_counts = emissions_df.groupby(['Industry_Category', 'Gas_Type']).size().reset_index(name='Count')

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(x='Count', y='Industry_Category', hue='Gas_Type', data=industry_gas_counts)
plt.title('Gas Types by Industry')
plt.xlabel('Count')
plt.ylabel('Industry')
plt.legend(title='Gas Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# List of top ten economy countries
top_ten_economy = [
    'United States', 'China, P.R.: Mainland', 'Japan', 'Germany', 'India',
    'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada'
]

# Filter the DataFrame to include only the top ten economy countries
df_top_ten = df[df['Country'].isin(top_ten_economy)]

# Select relevant columns for analysis (e.g., 'Country', 'F2016' to 'F2021')
columns_of_interest = ['Country', 'F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021', 'Gas_Type']
df_selected = df_top_ten[columns_of_interest]

# Filter rows where Gas_Type is 'Methane'
methane_data = df_selected[df_selected['Gas_Type'] == 'Methane']

# Drop 'Gas_Type' column as it's now redundant
methane_data = methane_data.drop('Gas_Type', axis=1)

# Group by 'Country' and take the mean for each year
methane_mean_by_country = methane_data.groupby('Country').mean()

# Transpose the DataFrame for easier plotting
methane_mean_by_country_T = methane_mean_by_country.T

# Plot the trend of mean methane emissions for each country from 2016 to 2021
plt.figure(figsize=(12, 8))
methane_mean_by_country_T.plot(kind='line', marker='o')
plt.title('Mean Methane Emissions Trend (2016 to 2021) - Top Ten Economy Countries')
plt.xlabel('Year')
plt.ylabel('Mean Methane Emissions (Million metric tons of CO2 equivalent)')
plt.legend(title='Country', bbox_to_anchor=(1, 1))
plt.show()


# %%
df.columns

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# Select relevant columns for analysis
columns_of_interest = ['Industry', 'Gas_Type', 'F2021','F2016','F2017','F2018','F2019','F2020']
df_selected = df[columns_of_interest].copy()

# Check for missing values
print(df_selected.isnull().sum())

# If there are missing values, handle them (e.g., fill with mean, median, or use imputation techniques)
df_selected = df_selected.dropna()  # For simplicity, let's drop rows with missing values

# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df_selected, columns=['Industry', 'Gas_Type'])

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X = df_encoded.drop('F2021', axis=1)  # Independent variables
y = df_encoded['F2021']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)
print(y_pred)
# Evaluate the model's performance (e.g., using Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Access coefficients to understand the impact of each feature
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# Select relevant columns for analysis (e.g., 'Country', 'F2021', 'Industry', 'Gas_Type', etc.)
columns_of_interest = ['Country', 'F2021', 'Industry', 'Gas_Type']
df_selected = df[columns_of_interest]

# Calculate total emissions for each country in 2021
total_emissions_2021 = df_selected.groupby('Country')['F2021'].sum()

# Identify the top and bottom 10%
top_10_percent = total_emissions_2021.nlargest(int(0.1 * len(total_emissions_2021)))
bottom_10_percent = total_emissions_2021.nsmallest(int(0.1 * len(total_emissions_2021)))

# Plot for the top 10%
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_percent.index, y=top_10_percent)
plt.title('Top 10%: Total Emissions in 2021')
plt.xlabel('Country')
plt.ylabel('Total Emissions in 2021 (Million metric tons of CO2 equivalent)')
plt.xticks(rotation=45, ha='right')
plt.show()

# Plot for the bottom 10%
plt.figure(figsize=(12, 8))
sns.barplot(x=bottom_10_percent.index, y=bottom_10_percent)
plt.title('Bottom 10%: Total Emissions in 2021')
plt.xlabel('Country')
plt.ylabel('Total Emissions in 2021 (Million metric tons of CO2 equivalent)')
plt.xticks(rotation=45, ha='right')
plt.show()

#%%
print(top_10_percent.index)
print(bottom_10_percent.index)



# %%
from scipy.stats import ttest_ind

# Example: Compare emissions in 2021 between top and bottom groups
top_group_data = df_selected[df_selected['Country'].isin(top_10_percent.index)]['F2021']
bottom_group_data = df_selected[df_selected['Country'].isin(bottom_10_percent.index)]['F2021']

# Perform independent t-test
t_stat, p_value = ttest_ind(top_group_data, bottom_group_data)

# Analyze results
if p_value < 0.05:
    print("There is a significant difference between the top and bottom groups.")
else:
    print("There is no significant difference between the top and bottom groups.")

# %%
from scipy.stats import chi2_contingency

# Example: Compare the distribution of industries between top and bottom groups
industry_contingency_table = pd.crosstab(df_selected['Industry'], df_selected['Country'].isin(top_10_percent.index))

# Perform chi-square test
chi2_stat, p_value, _, _ = chi2_contingency(industry_contingency_table)

# Analyze results
if p_value < 0.05:
    print("There is a significant difference in the distribution of industries between the top and bottom groups.")
else:
    print("There is no significant difference in the distribution of industries between the top and bottom groups.")


# %%




# %%
df.columns
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# List of countries to focus on
countries_of_interest_top = [
    'China, P.R.: Mainland', 'United States', 'India', 'Russian Federation',
    'Japan', 'Indonesia', 'Iran, Islamic Rep. of', 'Brazil', 'Saudi Arabia'
]

# Filter the DataFrame to include only the selected countries
df_filtered = df[df['Country'].isin(countries_of_interest_top)]

# Select relevant columns for analysis (e.g., 'Country', 'F2021', 'Industry', etc.)
columns_of_interest = ['Country', 'F2021', 'Industry_Category']
df_selected = df_filtered[columns_of_interest]

# Group by Country and Industry, summing up emissions for each combination
df_grouped = df_selected.groupby(['Country', 'Industry_Category'])['F2021'].sum().reset_index()

# Set the plot size
plt.figure(figsize=(14, 8))

# Use seaborn to create a bar plot
sns.barplot(x='Country', y='F2021', hue='Industry_Category', data=df_grouped, palette="tab10")

# Add labels and title
plt.title('Emissions by Industry in 2021 for Each Country')
plt.xlabel('Country')
plt.ylabel('Total Emissions')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show legend
plt.legend(title='Industry_Category', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# List of countries to focus on
countries_of_interest_bottom = [
    'Macao', 'Iceland', 'Cyprus', 'Luxembourg', 'Latvia',
       'Estonia, Rep. of', 'El Salvador', 'Costa Rica', 'Papua New Guinea'
]

# Filter the DataFrame to include only the selected countries
df_filtered = df[df['Country'].isin(countries_of_interest_bottom)]

# Select relevant columns for analysis (e.g., 'Country', 'F2021', 'Industry', etc.)
columns_of_interest = ['Country', 'F2021', 'Industry_Category']
df_selected = df_filtered[columns_of_interest]

# Group by Country and Industry, summing up emissions for each combination
df_grouped = df_selected.groupby(['Country', 'Industry_Category'])['F2021'].sum().reset_index()

# Set the plot size
plt.figure(figsize=(14, 8))

# Use seaborn to create a bar plot
sns.barplot(x='Country', y='F2021', hue='Industry_Category', data=df_grouped, palette="tab10")

# Add labels and title
plt.title('Emissions by Industry in 2021 for Each Country')
plt.xlabel('Country')
plt.ylabel('Total Emissions')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show legend
plt.legend(title='Industry_Category', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# List of countries to focus on
countries_of_interest_top = [
    'China, P.R.: Mainland', 'United States', 'India', 'Russian Federation',
    'Japan', 'Indonesia', 'Iran, Islamic Rep. of', 'Brazil', 'Saudi Arabia'
]

# Filter the DataFrame to include only the selected countries
df_filtered = df[df['Country'].isin(countries_of_interest_top)]

# Select relevant columns for analysis (e.g., 'Country', 'F2021', 'Gas_Type', etc.)
columns_of_interest = ['Country', 'F2021', 'Gas_Type']
df_selected = df_filtered[columns_of_interest]

# Group by Country and Gas_Type, summing up emissions for each combination
df_grouped = df_selected.groupby(['Country', 'Gas_Type'])['F2021'].sum().reset_index()

# Set the plot size
plt.figure(figsize=(14, 8))

# Use seaborn to create a bar plot
sns.barplot(x='Country', y='F2021', hue='Gas_Type', data=df_grouped, palette="tab10")

# Add labels and title
plt.title('Emissions by Gas Type in 2021 for Each Country')
plt.xlabel('Country')
plt.ylabel('Total Emissions')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show legend
plt.legend(title='Gas_Type', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named df
# Replace 'your_data.csv' with the actual path or URL to your CSV file if needed
# df = pd.read_csv('your_data.csv')

# List of countries to focus on
countries_of_interest_bottom = [
    'Macao', 'Iceland', 'Cyprus', 'Luxembourg', 'Latvia',
    'Estonia, Rep. of', 'El Salvador', 'Costa Rica', 'Papua New Guinea'
]

# Filter the DataFrame to include only the selected countries
df_filtered = df[df['Country'].isin(countries_of_interest_bottom)]

# Select relevant columns for analysis (e.g., 'Country', 'F2021', 'Gas_Type', etc.)
columns_of_interest = ['Country', 'F2021', 'Gas_Type']
df_selected = df_filtered[columns_of_interest]

# Group by Country and Gas_Type, summing up emissions for each combination
df_grouped = df_selected.groupby(['Country', 'Gas_Type'])['F2021'].sum().reset_index()

# Set the plot size
plt.figure(figsize=(14, 8))

# Use seaborn to create a bar plot
sns.barplot(x='Country', y='F2021', hue='Gas_Type', data=df_grouped, palette="tab10")

# Add labels and title
plt.title('Emissions by Gas Type in 2021 for Each Country (Bottom 10%)')
plt.xlabel('Country')
plt.ylabel('Total Emissions')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show legend
plt.legend(title='Gas_Type', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()


# %%
df['Gas_Type'].unique()
# %%
