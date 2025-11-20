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
plt.hist(area_df_clean['TotalArea'], bins=10, edgecolor = 'black')
plt.xlabel("Total Area")
plt.ylabel("Frequency")
plt.title("Total Area of real estate in Portugal")

filter = (data['Price'] < 1500000) & (data['Price'] > 10000)
df_clean = data[filter]
plt.hist(df_clean['Price'] / 1000, bins=50, edgecolor = 'black')
plt.ticklabel_format(style='plain', axis='x')
plt.title('Price Houses in Portugal')
plt.xlabel('Preço em milhares de €')
plt.ylabel('Frequência')
ax = plt.gca()
ax.format_coord = lambda x, y: f"x={x:.0f} y={y:.0f}"
plt.show()

#print(data.info())


#TotalArea
#ConstructionYear




