"""
Real Estate Price Prediction - Data Cleaning & EDA
This script loads the Portugal Real Estate dataset, handles missing values using
SimpleImputer, and performs initial Exploratory Data Analysis (EDA).
Made by mankz111
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 2. LOAD DATA
# ---------------------------------------------------------
# Using os.path.join ensures the path works on both Windows and Mac/Linux
csv = os.path.join(os.path.dirname(__file__), 'portugal_real_estate.csv')
data = pd.read_csv(csv, low_memory=False)

# 3. DATA CLEANING: MISSING VALUES
# ---------------------------------------------------------

#Calculated the missing percentage on several values of the dataset
# missing_percentage = (data.isnull().sum() / len(data)) * 100
# print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

#Used imputer to fix the missing values on the cols
imputer = SimpleImputer(strategy='median')
cols_to_fix = ['LivingArea', 'TotalArea', 'NumberOfBathrooms']

data[cols_to_fix] = imputer.fit_transform(data[cols_to_fix])
print("Missing Values:", data[cols_to_fix].isnull().sum().sum())

# 4. EXPLORATORY DATA ANALYSIS (EDA) & OUTLIER DETECTION
# ---------------------------------------------------------
# print(data['TotalArea'].describe())

# Defining filters to remove outliers for visualization purposes
# Rule: Remove physical impossibilities (Area < 10) and extreme outliers
# clean_area = (data['TotalArea'] > 10) & (data['TotalArea'] < 5000)
# area_df_clean = data[clean_area]

# Rule: Focus on the most representative price range (10k to 1.5M)
# filter_price = (data['Price'] < 1500000) & (data['Price'] > 10000)
# df_clean = data[filter_price]

# 5. VISUALIZATION
# ---------------------------------------------------------
# Setup the figure with 2 subplots side-by-side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Distribution of Total Area
# ax1.hist(area_df_clean['TotalArea'], bins=100, edgecolor='black')
# ax1.set_xlabel("Total Area (m²)")
# ax1.set_ylabel("Frequency")
# ax1.set_title("Distribution: Total Area of Real Estate")

# Plot 2: Distribution of Price
# ax2.hist(df_clean['Price'], bins=50, edgecolor='black')
# ax2.set_xlabel("Price (EUR)")
# ax2.set_ylabel("Frequency")
# ax2.set_title("Distribution: House Prices in Portugal")

# plt.tight_layout() # Adjust spacing between plots
# plt.show()

# print(data.info())



