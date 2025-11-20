from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import streamlit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

csv = os.path.join(os.path.dirname(__file__), 'portugal_real_estate.csv')
data = pd.read_csv(csv, low_memory=False)

#print(data['TotalArea'].describe())

clean_area = (data['TotalArea'] > 10) & (data['TotalArea']<5000)
area_df_clean = data[clean_area]
filter = (data['Price'] < 1500000) & (data['Price'] > 10000)
df_clean = data[filter]

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
#Gráfico 1
ax1.hist(area_df_clean['TotalArea'], bins=10, edgecolor='black')
ax1.set_xlabel("Total Area")
ax1.set_ylabel("Frequency")
ax1.set_title("Total Area of real estate in Portugal")
#Gráfico 2

ax2.hist(df_clean['Price'], bins=10, edgecolor='black')
ax2.set_xlabel("Price")
ax2.set_ylabel("Frequency")
ax2.set_title("Price of Houses in Portugal")
plt.show()


#print(data.info())


#TotalArea
#ConstructionYear




