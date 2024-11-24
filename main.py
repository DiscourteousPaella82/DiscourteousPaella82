import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

import matplotlib as mpl


#Mace Head Weather Station daily data from 1st October 2024 to 30th October 2024
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

if __name__ == '__main__':
 main()