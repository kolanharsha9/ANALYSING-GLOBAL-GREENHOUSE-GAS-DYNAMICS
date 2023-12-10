# %%
# Outlier Detection in Greenhouse Gas Emission Changes
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# %%
# Load the dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv')

# %%
# Calculate the percentage change in emissions for outlier detection
# This helps in identifying significant increases or decreases in emissions year-over-year
df['Emission_Change'] = df['F2021'].pct_change()

# %%
# Visualizing potential outliers using a boxplot
# Boxplots are effective for spotting outliers in data
sns.boxplot(x=df['Emission_Change'])
plt.title('Boxplot for Emission Change')
plt.show()

# %%
# Outlier Detection using the Z-Score method
# The Z-Score is a measure of how many standard deviations a data point is from the mean
df['Emission_Change_ZScore'] = stats.zscore(df['Emission_Change'])
df['Outlier_ZScore'] = (df['Emission_Change_ZScore'] > 3) | (df['Emission_Change_ZScore'] < -3)
print("Potential Outliers identified by Z-Score:")
print(df[df['Outlier_ZScore']])

# %%
# Outlier Detection using the IQR method
# The IQR method identifies outliers based on the quartile range of the data
Q1 = df['Emission_Change'].quantile(0.25)
Q3 = df['Emission_Change'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_upper = Q3 + 1.5 * IQR
outlier_threshold_lower = Q1 - 1.5 * IQR
df['Outlier_IQR'] = (df['Emission_Change'] < outlier_threshold_lower) | (df['Emission_Change'] > outlier_threshold_upper)
print("Potential Outliers identified by IQR:")
print(df[df['Outlier_IQR']])

# %%

# %%

#%%
# Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load the dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv')

#%%
# Basic statistics
print(df.describe())

#%%
# Check for missing values
print(df.isnull().sum())

#%%
# Visualize the distribution of emissions for the year 2021
plt.figure(figsize=(10, 6))
sns.histplot(df['F2021'], bins=30, kde=True)
plt.title('Distribution of GHG Emissions in 2021')
plt.xlabel('GHG Emissions')
plt.ylabel('Frequency')
plt.show()

#%%
# Compare emissions in different industries
plt.figure(figsize=(12, 6))
sns.barplot(x='Industry', y='F2021', data=df)
plt.xticks(rotation=45)
plt.title('GHG Emissions by Industry in 2021')
plt.xlabel('Industry')
plt.ylabel('GHG Emissions')
plt.show()

#%%
# Compare emissions in different countries (10 countries)
sample_countries = df['Country'].drop_duplicates().sample(10)
df_sample = df[df['Country'].isin(sample_countries)]
plt.figure(figsize=(12, 6))
sns.barplot(x='Country', y='F2021', data=df_sample)
plt.xticks(rotation=45)
plt.title('GHG Emissions by Country in 2021 (Sample)')
plt.xlabel('Country')
plt.ylabel('GHG Emissions')
plt.show()

#%%

#%%

#%%
# Linear Regression Modeling

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#%%
# Load the dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv') 

#%%
# Selecting a few columns for simplicity - replace with relevant columns
selected_columns = ['F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']  # Example columns
df_selected = df[selected_columns].dropna()  # Drop rows with missing values

#%%
# Prepare the data
X = df_selected.drop('F2021', axis=1)  # Features (e.g., emissions from 2016 to 2020)
y = df_selected['F2021']               # Target variable (e.g., emissions in 2021)

#%%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#%%
# Make predictions
y_pred = model.predict(X_test)

#%%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#%%
# Print out the model performance metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#%%

#%%

#%%
# Decision Tree Regression Modeling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

#%%
# Load the dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv') 

#%%
# Selecting a few columns for simplicity - replace with relevant columns
selected_columns = ['F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021'] 
df_selected = df[selected_columns].dropna() 

#%%
# Prepare the data
X = df_selected.drop('F2021', axis=1) 
y = df_selected['F2021']  

#%%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
# Create and train the decision tree regressor model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#%%
# Make predictions
y_pred = model.predict(X_test)

#%%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# %%
# Print out the model performance metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# %%

# %%

# %%
# Random Forest Regression for Predicting Greenhouse Gas Emissions
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %%
# Load the dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv') 

# %%
# Selecting columns for the model
selected_columns = ['F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']
df_selected = df[selected_columns].dropna()

# %%
# Prepare the data
X = df_selected.drop('F2021', axis=1) 
y = df_selected['F2021']       

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Create and train the random forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # n_estimators can be adjusted
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)

# %%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# %%
# Print out the model performance metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# %%

# %%

# %%
# Time Series Analysis of Methane Emissions in the United States

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
# Load the dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv') 

# %%
# Filter data for the United States and Methane emissions
filtered_df = df[(df['Country'] == 'United States') & (df['Gas_Type'] == 'Methane')]

# %%
# Creating a time series of Methane emissions for the United States
emissions_data = filtered_df[['F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']].mean()

# %%
# Time Series Plot
emissions_data.plot()
plt.title('Yearly Methane Emissions in the United States')
plt.xlabel('Year')
plt.ylabel('Emissions')
plt.xticks(range(len(emissions_data.index)), emissions_data.index, rotation=45)
plt.show()

# %%
# ARIMA Model
# Note: The order parameters might need to be adjusted based on the data
model = ARIMA(emissions_data, order=(1, 1, 1))
model_fit = model.fit()

# %%
# Summary of the model
print(model_fit.summary())

# %%
# Plot ACF and PACF
plot_acf(emissions_data, lags=2) 
plot_pacf(emissions_data, lags=2)  
plt.show()

# %%

# %%

# %%
# Cluster Analysis of Greenhouse Gas Emissions by Country

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %%
# Load dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv') 

# %%
selected_columns = ['F2016', 'F2017', 'F2018', 'F2019', 'F2020', 'F2021']
df_selected = df[selected_columns].apply(pd.to_numeric, errors='coerce')

# %%
df_selected = df_selected.dropna()

# %%
df_selected['Country'] = df['Country']

# %%
# Aggregate emissions data for clustering
emissions_data = df_selected.groupby('Country').mean()

# %%
# KMeans Clustering
kmeans = KMeans(n_clusters=3) 
clusters = kmeans.fit_predict(emissions_data)

# %%
# Add cluster info to the dataframe
emissions_data['Cluster'] = clusters

# %%
# Plot the clusters
plt.scatter(emissions_data['F2020'], emissions_data['F2021'], c=emissions_data['Cluster'])
plt.xlabel('Emissions in 2020')
plt.ylabel('Emissions in 2021')
plt.title('Cluster of Countries based on GHG Emissions')
plt.show()

# %%

# %%

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# %%
# Load dataset
df = pd.read_csv('/Users/ericazhao/Documents/GitHub/Intro-to-Data-Mining-Project/data.csv') 

# %%
# Categorizing countries into high and low emission based on F2021 emissions
# Create a new column 'Emission_Category' for classification
median_emissions = df['F2021'].median()
df['Emission_Category'] = np.where(df['F2021'] > median_emissions, 'High', 'Low')

# %%
# Prepare the data for classification
# Selecting relevant features for the model
features = ['F2016', 'F2017', 'F2018', 'F2019', 'F2020'] 
X = df[features]
y = df['Emission_Category']

# %%
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Creating and training the SVM model
model = SVC(kernel='linear')  
model.fit(X_train, y_train)

# %%
# Making predictions
y_pred = model.predict(X_test)

# %%
# Evaluating the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%

