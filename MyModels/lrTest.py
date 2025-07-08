from models.MyLogRegression import MyLogRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

data = pd.read_csv("datasets/Valorant5.csv")

np_data = data.drop(["Serge_peak"], axis=1).to_numpy()

data['avg_kd'] = data[['Vlad_kd', 'Serge_kd', 'Dima_kd']].mean(axis=1)

# Кодируем выбор персонажа
encoder = OneHotEncoder()
peak_encoded = encoder.fit_transform(data[['Vlad_peak', 'Dima_peak']]).toarray()

# Объединяем с остальными фичами
X = np.hstack([data[['Vlad_kd', 'Serge_kd', 'Dima_kd', 'avg_kd']].values, peak_encoded])


#X = np.array(data[['Vlad_kd', 'Serge_kd', 'Dima_kd', 'avg_kd']].values)
y = np.array(data['Won'].values)

model = MyLogRegression()

model.fit(X=X,y=y)
print(model.get_w())
