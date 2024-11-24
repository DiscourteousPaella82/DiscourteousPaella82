import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
import numpy as np


#Mace Head Weather Station daily data from 1st October 2024 to 31st October 2024, date not significant
#model needs to predict maximum air temperature given features

def main():
    weather_train = pd.read_csv("dly275.csv",
     names=["Maximum Air Temperature", "Minimum  Air Temperature", "09utc Grass Minimum Temperature", "Mean 10cm soil temperature",
               "Mean CBL Pressure", "Mean Wind Speed", "Highest ten minute mean wind speed", "Wind Direction at max 10 min mean", "Highest Gust",
            "Potential Evapotranspiration", "Evaporation", "Soil Moisture Deficits(mm) well drained",
            "Soil Moisture Deficits(mm) moderately drained", "Soil Moisture Deficits(mm) poorly drained", "Global Radiation"])

    weather_train.head()

    weather_features = weather_train.copy()
    weather_labels = weather_features.pop('Maximum Air Temperature')

    weather_features = np.array(weather_features)
    weather_features

    weather_model = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    weather_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=tf.keras.optimizers.Adam())

    weather_model.fit(weather_features, weather_labels, epochs=10)

    normalize = layers.Normalization()

    #normalized model
    normalize.adapt(weather_features)

    norm_weather_model = tf.keras.Sequential([
        normalize,
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    norm_weather_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam())

    norm_weather_model.fit(weather_features, weather_labels, epochs=10)

if __name__ == '__main__':
 main()