import pandas as pd
from helping_functions import *

weather_df = pd.read_csv("Tulcea.csv")
weather_df.drop(["feelslikemax", "feelslikemin", "dew", "precipprob","precipcover", "snowdepth", "windgust",
                        "cloudcover", "visibility", "solarradiation", "solarenergy", "severerisk", "moonphase", "conditions", "icon", "stations", "description", "temp"],
                axis='columns', inplace=True)
weather_df.rename(columns={"datetime" : "date"}, inplace=True)

columns_to_convert_to_celsius = ["tempmax"]
# weather_df[columns_to_convert_to_celsius] = weather_df.apply(lambda x: convert_fahrenheit_to_celsius(x))
weather_df[["tempmax", "tempmin", "feelslike"]] = weather_df[["tempmax", "tempmin", "feelslike"]].map(convert_fahrenheit_to_celsius)


print(weather_df.columns)
print(weather_df.loc[1, "name"])
print(weather_df.loc[1, "date"])
print(weather_df.loc[1, "tempmax"])
print(weather_df.loc[1, "tempmin"])
print(weather_df.loc[1, "feelslike"])
print(weather_df.loc[1, "humidity"])
print(weather_df.loc[1, "precip"])
print(weather_df.loc[1, "snow"])
print(weather_df.loc[1, "windspeed"])
print(weather_df.loc[1, "uvindex"])
print(weather_df.loc[1, "sunrise"])
print(weather_df.loc[1, "sunset"])
print(weather_df.loc[1, "winddir"])
print(weather_df.loc[1, "preciptype"])
print(weather_df.loc[1, "sealevelpressure"])




