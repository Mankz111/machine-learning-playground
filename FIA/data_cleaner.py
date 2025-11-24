from sklearn.impute import SimpleImputer
import pandas as pd
import os


try:
    csv = os.path.join(os.path.dirname(__file__), 'portugal_real_estate.csv')
except:
    FileNotFoundError
data = pd.read_csv(csv, low_memory=False)



missing_percentage = (data.isnull().sum() / len(data)) * 100
cols_to_drop = ['ConservationStatus', 'BuiltArea', 'GrossArea', 'Floor', 'PublishDate', 'LotSize', 'NumberOfBedrooms', 'NumberOfWC', 'EnergyEfficiencyLevel', 'ConstructionYear', 'ElectricCarsCharging', 'TotalRooms', 'LivingArea']
new_data = data.drop(cols_to_drop, axis=1)

missing_percentage_updated = (new_data.isnull().sum() / len(data)) * 100
data_to_put_zero = ['Garage', 'HasParking']
final_data = new_data.copy()
final_data[data_to_put_zero] = final_data[data_to_put_zero].fillna(0)




# imputer = SimpleImputer(strategy='median')
# print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))
# print(missing_percentage_updated[missing_percentage_updated>0].sort_values(ascending=False))
# print(final_data.info())
print(final_data['HasParking'].unique())
print(final_data['Garage'].unique())
 
    


    





