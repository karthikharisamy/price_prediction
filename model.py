import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import joblib
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 16, 10


df = pd.read_csv('C:/Users/SARAVANAN HARI/Desktop/Price Prediction/NYC_2019.csv')

df.shape
df.columns
sns.distplot(df.price);


sns.distplot(np.log1p(df.price));


sns.countplot(x='neighbourhood_group', data=df);

corr_matrix = df.corr()


price_corr = corr_matrix['price']
price_corr.iloc[price_corr.abs().argsort()]


palette = sns.diverging_palette(20, 220, n=256)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=palette, vmax=.3, center=0,
            square=True, linewidths=.5);


missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

df = df.drop(['id', 'name', 'host_id', 'host_name', 'reviews_per_month', 'last_review', 'neighbourhood'], axis=1)


missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)


df.columns

X = df.drop('price', axis=1)
y = np.log1p(df.price.values)


from sklearn.preprocessing import OneHotEncoder

data = [['Manhattan'], ['Brooklyn']]

OneHotEncoder(sparse=False).fit_transform(data)




from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

transformer = make_column_transformer(
    (MinMaxScaler(), ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']),
    (OneHotEncoder(handle_unknown="ignore"), ['neighbourhood_group', 'room_type'])
)


transformer.fit(X)


X = transformer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def plot_mse(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.plot(hist['epoch'], hist['mse'],
            label='Train MSE')
  plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val MSE')
  plt.legend()
  plt.show()
  
  
  
model = keras.Sequential()
model.add(keras.layers.Dense(units=64, activation="relu", input_shape=[X_train.shape[1]]))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(units=32, activation="relu"))
model.add(keras.layers.Dropout(rate=0.5))

model.add(keras.layers.Dense(1))

model.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss = 'mse', 
    metrics = ['mse'])

BATCH_SIZE = 32

early_stop = keras.callbacks.EarlyStopping(
  monitor='val_mse',
  mode="min",
  patience=10
)

history = model.fit(
  x=X_train,
  y=y_train,
  shuffle=True,
  epochs=100,
  validation_split=0.2,
  batch_size=BATCH_SIZE,
  callbacks=[early_stop]
)

plot_mse(history)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score


y_pred = model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
Rmse=np.sqrt(mean_squared_error(y_test, y_pred))
R_Sqaure= r2_score(y_test, y_pred)


joblib.dump(transformer, "data_transformer.joblib")
model.save("price_prediction_model.h5")