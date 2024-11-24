import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


"""Using TensorFlow in attempt to make predictions of Maximum Air Temperature 

Mace Head Weather Station daily data from 1st October 2024 to 31st October 2024, date not significant
model attempts to predict maximum air temperature given the other features
dataset from https://cli.fusio.net/cli/climate_data/webdata/dly275.csv
date and index columns were removed from the dataset in use
14 features"""

def main():
    weather_train = pd.read_csv("dly275.csv",
     names=["Minimum  Air Temperature", "09utc Grass Minimum Temperature", "Mean 10cm soil temperature",
               "Mean CBL Pressure", "Mean Wind Speed", "Highest ten minute mean wind speed", "Wind Direction at max 10 min mean", "Highest Gust",
            "Potential Evapotranspiration", "Evaporation", "Soil Moisture Deficits(mm) well drained",
            "Soil Moisture Deficits(mm) moderately drained", "Soil Moisture Deficits(mm) poorly drained", "Global Radiation", "Maximum Air Temperature"])

    weather_train.head()

    weather_features = weather_train.copy()
    weather_labels = weather_features.pop("Maximum Air Temperature")    #Removing 'Maximum Air Temperature' from the features data frame and pushing them to a label data frame

    weather_features = np.array(weather_features)

    weather_model = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(1) #1 output unit
    ])

    weather_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=tf.keras.optimizers.Adam())

    weather_model.fit(weather_features, weather_labels, epochs=50)

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

    #sample weather data from 30th September 2024 (10.2,9.8,0.5,1007.8,11.8,19,300,25,1.4,2.2,34.3,34.3,32.1,1282)

    norm_weather_model.fit(weather_features, weather_labels, epochs=50)
    sample_weather = np.array([[10.2,9.8,0.5,1007.8,11.8,19,300,25,1.4,2.2,34.3,34.3,32.1,1282]])

    print(f"Feature predictions:\n{norm_weather_model.predict(weather_features)}")
    print(f"Sample prediction:{norm_weather_model.predict(sample_weather)}")

if __name__ == '__main__':
 main()