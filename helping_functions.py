from datetime import datetime

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