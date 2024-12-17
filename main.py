import pandas as pd
import numpy as np
from helping_functions import *

weather_df = pd.read_csv("Tulcea 2023-01-01 to 2024-12-01.csv")
weather_df.drop(["feelslikemax", "feelslikemin", "dew", "precipprob","precipcover", "snowdepth", "windgust",
                        "visibility", "solarradiation", "solarenergy", "severerisk", "moonphase", "conditions", "icon", "stations", "description", "temp"],
                axis='columns', inplace=True)
weather_df.rename(columns={"datetime" : "date"}, inplace=True)

# print(weather_df.columns)
# print(weather_df.loc[1, "name"])
# print(weather_df.loc[1, "date"])
# print(weather_df.loc[1, "tempmax"])
# print(weather_df.loc[1, "tempmin"])
# print(weather_df.loc[1, "feelslike"])
# print(weather_df.loc[1, "humidity"])
# print(weather_df.loc[1, "precip"])
# print(weather_df.loc[1, "windspeed"])
# print(weather_df.loc[1, "uvindex"])
# print(weather_df.loc[1, "winddir"])
# print(weather_df.loc[1, "preciptype"])
# print(weather_df.loc[1, "sealevelpressure"])
# print(weather_df.loc[1, "cloudcover"])

weather_df.sort_values(by='date', inplace=True)

weather_df['season'] = weather_df['date'].apply(get_season)
weather_df['temp_state'] = weather_df['tempmax'].apply(classify_state)
weather_df['state_season'] = weather_df['temp_state'] + "_" + weather_df['season']

temp_states = ["frig", "moderat", "cald"]
seasons = ["iarna", "primavara", "vara", "toamna"]
all_states = [f"{t}_{s}" for s in seasons for t in temp_states]

transition_matrix = pd.DataFrame(0, index=all_states, columns=all_states, dtype=float)

states_sequence = weather_df['state_season'].tolist()

for i in range(len(states_sequence)-1):
    current_state = states_sequence[i]
    next_state = states_sequence[i+1]
    transition_matrix.loc[current_state, next_state] += 1

transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

print("Matricea de tranziție:")
print(transition_matrix)
print()

def get_state_season_for_date(df, date_str):
    day_data = df.loc[df['date'] == date_str]
    if not day_data.empty:
        return day_data.iloc[0]['state_season']
    else:
        raise ValueError("Data specificată nu există în DataFrame")

def state_to_distribution(state, all_states):
    dist = [0]*len(all_states)
    dist[all_states.index(state)] = 1
    return np.array(dist)

selected_date = "2024-12-01"
current_state_season = get_state_season_for_date(weather_df, selected_date)
current_distribution = state_to_distribution(current_state_season, all_states)

print(f"Starea sezonieră pentru {selected_date} este '{current_state_season}'")
print("Distribuția curentă:", current_distribution)
print()

next_day_prob = current_distribution.dot(transition_matrix.values)
print(f"Probabilitățile pentru ziua următoare după {selected_date}:")
for s, p in zip(all_states, next_day_prob):
    print(f"{s}: {p:.2f}")

print()

days_ahead = 20
future_distribution = current_distribution.copy()
for i in range(days_ahead):
    future_distribution = future_distribution.dot(transition_matrix.values)

print(f"Probabilitățile peste {days_ahead} zile de la {selected_date}:")
for s, p in zip(all_states, future_distribution):
    print(f"{s}: {p:.2f}")