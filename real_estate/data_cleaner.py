from sklearn.impute import SimpleImputer
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

try:
    csv = os.path.join(os.path.dirname(__file__), 'portugal_real_estate.csv')
except:
    FileNotFoundError
data = pd.read_csv(csv, low_memory=False)



missing_percentage = (data.isnull().sum() / len(data)) * 100
cols_to_drop = ['ConservationStatus', 'BuiltArea', 'GrossArea', 'Floor', 'PublishDate', 'LotSize', 'NumberOfBedrooms', 'NumberOfWC', 'EnergyEfficiencyLevel', 'ConstructionYear', 'ElectricCarsCharging', 'TotalRooms', 'LivingArea', 'Parking']
new_data = data.drop(cols_to_drop, axis=1)
missing_percentage_updated = (new_data.isnull().sum() / len(data)) * 100
data_to_put_zero = ['Garage', 'HasParking']
final_data = new_data.copy()
final_data[data_to_put_zero] = final_data[data_to_put_zero].fillna(0)
encoder = LabelEncoder()
hasParkingCat = final_data['HasParking']
final_data['HasParking'] = encoder.fit_transform(hasParkingCat)
hasGarage = final_data['Garage']
final_data['Garage'] = encoder.fit_transform(hasGarage)
clean_area = (final_data['TotalArea'] > 20) & (final_data['TotalArea'] < 2000)
clean_price = (final_data['Price'] > 10000) & (final_data['Price'] < 5000000)
clean_bath = (final_data['NumberOfBathrooms'] > 0)
final_data = final_data[clean_area & clean_price & clean_bath]
final_data['Elevator'] = final_data['Elevator'].astype(int)

model = LinearRegression





#print(final_data.info())
# print(final_data['Type'].unique())
# print(final_data['EnergyCertificate'].unique())
# print(final_data['NumberOfBathrooms'].describe())
# print(final_data['HasParking'], final_data['Garage'])
# print(final_data['Price'].describe())
# print(final_data['TotalArea'].describe())
# print(final_data['Elevator'].describe())
# imputer = SimpleImputer(strategy='median')
# print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))
# print(missing_percentage_updated[missing_percentage_updated>0].sort_values(ascending=False))
# print(final_data.info())
# print(final_data['HasParking'].unique())
# print(final_data['Garage'].unique())



    





