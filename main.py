import pymc as pm
import arviz as az
import pandas as pd
import pickle

def predict_feelslike(test_data, trace_path="model_trace.nc", scaler_path="scaler.pkl"):
    print("=== Încărcăm trace-ul și scaler-ul ===")
    trace = az.from_netcdf(trace_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print("=== Normalizăm datele de test ===")
    normalized_data = scaler.transform(test_data[["tempmax", "humidity", "windspeed"]])

    print("=== Recreăm modelul pentru predicții ===")
    with pm.Model() as model:
        tempmax_data = pm.Data("tempmax_data", normalized_data[:, 0])
        humidity_data = pm.Data("humidity_data", normalized_data[:, 1])
        windspeed_data = pm.Data("windspeed_data", normalized_data[:, 2])

        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta_temp = pm.Normal("beta_temp", mu=0, sigma=0.3)
        beta_hum = pm.Normal("beta_hum", mu=0, sigma=0.3)
        beta_wind = pm.Normal("beta_wind", mu=0, sigma=0.3)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        mu = alpha + beta_temp*tempmax_data + beta_hum*humidity_data + beta_wind*windspeed_data

        feelslike_obs = pm.Normal("feelslike_obs", mu=mu, sigma=sigma, observed=None)

        print("=== Generăm predicții ===")
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=["feelslike_obs"],
            return_inferencedata=False
        )

        arr = ppc["feelslike_obs"]  # Forma e (chains, draws, n_test)
        print("Forma lui ppc['feelslike_obs'] =", arr.shape)

        # Facem media pe chain și draws => rămâne axa test (3 rânduri)
        predictions = arr.mean(axis=(0, 1))

    return pd.Series(predictions, index=test_data.index)


if __name__ == "__main__":
    test_data = pd.DataFrame({
        "tempmax": [30],
        "humidity": [70],
        "windspeed": [10]
    })

    predicted_feelslike = predict_feelslike(test_data)
    print("=== Predicții pentru datele de test ===")
    print(predicted_feelslike)
