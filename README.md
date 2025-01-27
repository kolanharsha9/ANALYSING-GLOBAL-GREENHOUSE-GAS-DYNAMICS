# Global Greenhouse Gas Dynamics Analysis

## Project Overview
This project provides a comprehensive analysis of global greenhouse gas (GHG) emissions from 2016 to 2021. By utilizing datasets from the **United Nations Framework Convention on Climate Change (UNFCCC)** and the **International Monetary Fund (IMF)**, we investigate:

- Annual country-level GHG emissions across sectors.
- Key gas types and industry contributors to emissions.
- Emission trends in major economies.
- Methane mitigation efforts.
- Disparities in emissions between top and bottom emitters.

The findings highlight actionable insights for policymakers and stakeholders to address climate challenges.

---

## Key Features

### 1. Sectoral Analysis
- Identifies the **Energy sector** as the largest contributor to global GHG emissions.
- Highlights the dominance of **CO2 emissions** in energy-intensive industries.

### 2. Emission Trends in Major Economies
- Tracks trends for the **top 10 economies** (e.g., United States, China, Germany, India).
- Observes:
  - Continuous increase in emissions from **China**.
  - Declining emissions in **European nations** and variable trends in the **United States**.

### 3. Methane Mitigation
- Evaluates **methane emissions** and identifies countries achieving reductions from 2016 to 2021.

### 4. Emissions Disparity
- Compares the **top 10% highest emitters** to the **bottom 10% emitters**.
- Highlights factors influencing disparities, including economic and industrial profiles.

### 5. Future Emissions Projections
- Employs **ARIMA (AutoRegressive Integrated Moving Average)** models to forecast emissions from 2022 to 2032.
- Projects increasing emissions for countries like **China** under a business-as-usual scenario.

---

## Data Description
- **Source:** UNFCCC and IMF datasets.
- **Timeframe:** 2016–2021 for analysis (historical data available from 1970).
- **Coverage:** 215 countries, six sectors, and 14 sub-sectors.
- **Greenhouse Gases:** CO2, CH4, N2O, F-gases.
- **Unique Features:** Includes emissions data with and without Land-Use, Land-Use Change, and Forestry (LULUCF).
- **Projections:** Future emissions until 2030 under a business-as-usual scenario.

---

## Methodology

### 1. Data Preparation
- Cleaned dataset by removing irrelevant columns (e.g., ISO2).
- Addressed missing values for reliability.
- Focused analysis on **2016–2021**, avoiding incomplete projections for 2022–2030.

### 2. Exploratory Data Analysis (EDA)
- Visualized trends using Python libraries (e.g., bar plots, line graphs).
- Identified dominant gas types per sector.

### 3. Statistical Modeling
- **Linear Regression**:
  - Achieved an R-squared value of 0.997 for 2021 emissions prediction.
- **ARIMA**:
  - Forecasted future emissions for top emitters.
  - Used Autocorrelation and Partial Autocorrelation for model selection (ARIMA(1,1,1)).

### 4. Statistical Tests
- Conducted **t-tests** and **Kolmogorov-Smirnov tests** to compare top and bottom emitters in 2021.

---

## Results and Insights

1. **Sectoral Contributions**: The Energy sector is the largest GHG contributor, with CO2 as the dominant gas.
2. **Emission Trends**:
   - China shows consistent emission increases.
   - European nations demonstrate slight declines.
   - The U.S. shows a downward trajectory post-2019.
3. **Methane Mitigation**: Successful reductions observed in multiple economies from 2020 to 2021.
4. **Emissions Disparity**: Significant differences exist between top and bottom emitters, emphasizing unequal contributions.
5. **Projections**: ARIMA forecasts indicate a need for urgent policy interventions to alter the trajectory of emissions.


