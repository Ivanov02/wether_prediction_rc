import pymc as pm
import arviz as az
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def train_and_save_model(csv_path="Tulcea 2023-01-01 to 2024-12-01.csv",
                         output_nc="model_trace.nc",
                         scaler_path="scaler.pkl"):
    """
    Antrenează modelul Bayesian pentru predicția temperaturii resimțite `feelslike`.
    """
    print("=== 1) Încarcă datele ===")
    df = pd.read_csv(csv_path)
    df = df[['tempmax', 'humidity', 'windspeed', 'feelslike']].dropna()

    # Normalizează predictorii
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[['tempmax', 'humidity', 'windspeed']])
    df[['tempmax', 'humidity', 'windspeed']] = normalized_data

    print("=== 2) Split train / test (80-20) ===")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"   - train size = {len(train_df)}")
    print(f"   - test size  = {len(test_df)}\n")

    print("=== 3) Definim modelul Bayesian PyMC ===")
    with pm.Model() as model:
        # Variabile de intrare
        tempmax_data = pm.Data("tempmax_data", train_df["tempmax"])
        humidity_data = pm.Data("humidity_data", train_df["humidity"])
        windspeed_data = pm.Data("windspeed_data", train_df["windspeed"])

        # Definim prioruri mai stricte pentru coeficienți
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta_temp = pm.Normal("beta_temp", mu=0, sigma=0.3)
        beta_hum = pm.Normal("beta_hum", mu=0, sigma=0.3)
        beta_wind = pm.Normal("beta_wind", mu=0, sigma=0.3)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        # Relație liniară
        mu = (
            alpha +
            beta_temp * tempmax_data +
            beta_hum * humidity_data +
            beta_wind * windspeed_data
        )

        # Observații
        feelslike_obs = pm.Normal("feelslike_obs", mu=mu, sigma=sigma, observed=train_df["feelslike"])

        print("=== 4) Sampling MCMC ===")
        trace = pm.sample(
            draws=500,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.99,
            random_seed=42
        )
        print("=== MCMC s-a terminat! ===\n")

    print(f"=== Salvăm trace-ul în fișierul '{output_nc}' ===")
    az.to_netcdf(trace, output_nc)

    print(f"=== Salvăm scaler-ul în fișierul '{scaler_path}' ===")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("=== Model și scaler salvați cu succes! ===")
    return model, trace, scaler


if __name__ == "__main__":
    train_and_save_model()
