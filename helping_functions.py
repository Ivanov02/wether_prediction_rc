from datetime import datetime
import numpy as np

def convert_fahrenheit_to_celsius(values):
    return round(((values - 32) / 1.8), 2)

def classify_state(tempmax):
    if tempmax < 5:
        return "frig"
    elif 5 <= tempmax < 15:
        return "moderat"
    else:
        return "cald"

def get_season(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    month = dt.month
    if month in [12, 1, 2]:
        return "iarna"
    elif month in [3, 4, 5]:
        return "primavara"
    elif month in [6, 7, 8]:
        return "vara"
    else:
        return "toamna"

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