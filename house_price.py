"""1. House Price Prediction: linear regression on the California Housing dataset.

Problem:  predict the median house value of a district from its size, location,
          age and income data.
Dataset:  California Housing (1990 US census), 20,640 districts, 8 features.
Model:    linear regression. Fitted two ways: the exact least-squares solution
          (scikit-learn LinearRegression), and batch gradient descent written by hand
          so we can watch the error fall step by step.

Run:  python house_price.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import dataset, r, save_result

COLS = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
        "population", "households", "median_income", "median_house_value"]

# ---- load ----
df = pd.read_csv(dataset("cal_housing") / "CaliforniaHousing" / "cal_housing.data", header=None, names=COLS)
n_raw = len(df)

# ---- EDA ----
capped = (df.median_house_value >= 500_000).mean()
corr = df.corr(numeric_only=True)["median_house_value"].drop("median_house_value").sort_values(ascending=False)
west = df[df.longitude < -121.5].median_house_value.median()  # west of -121.5° (Bay Area, coast)
east = df[df.longitude >= -121.5].median_house_value.median()

# ---- preprocessing ----
# Values were capped at $500,001 in the census data; those rows don't show the true price, so drop them.
df = df[df.median_house_value < 500_000].copy()
df["rooms_per_household"] = df.total_rooms / df.households
df["bedrooms_per_room"] = df.total_bedrooms / df.total_rooms
df["people_per_household"] = df.population / df.households
FEATURES = ["longitude", "latitude", "housing_median_age", "median_income",
            "rooms_per_household", "bedrooms_per_room", "people_per_household"]
# Clip extreme ratios (a few districts are hotels or dorms) to the 1st–99th percentile.
for c in ["rooms_per_household", "bedrooms_per_room", "people_per_household"]:
    df[c] = df[c].clip(*df[c].quantile([0.01, 0.99]))

X = df[FEATURES].to_numpy()
y = df.median_house_value.to_numpy() / 100_000  # target in $100k
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

# ---- model 1: exact least squares ----
ols = LinearRegression().fit(X_tr, y_tr)
pred = ols.predict(X_te)
rmse = np.sqrt(mean_squared_error(y_te, pred)) * 100_000
r2 = r2_score(y_te, pred)
baseline_rmse = np.sqrt(mean_squared_error(y_te, np.full_like(y_te, y_tr.mean()))) * 100_000

# ---- model 2: the same linear model learned by batch gradient descent, written by hand ----
# Start from all-zero weights and step downhill on the mean squared error; it should reach the
# least-squares answer above.
w, b, lr = np.zeros(X_tr.shape[1]), 0.0, 0.1
train_mse, val_mse = [], []
for step in range(40):
    err = X_tr @ w + b - y_tr
    w -= lr * 2 * X_tr.T @ err / len(y_tr)
    b -= lr * 2 * err.mean()
    train_mse.append(mean_squared_error(y_tr, X_tr @ w + b))
    val_mse.append(mean_squared_error(y_te, X_te @ w + b))
gd_rmse = np.sqrt(val_mse[-1]) * 100_000

coef = sorted(zip(FEATURES, ols.coef_), key=lambda t: -abs(t[1]))

print(f"rows {n_raw} -> {len(df)} after dropping capped values")
print(f"test RMSE ${rmse:,.0f}  R² {r2:.3f}  (predict-the-mean baseline ${baseline_rmse:,.0f})")
print(f"gradient descent after 40 steps: test RMSE ${gd_rmse:,.0f}")
print("largest coefficients:", [(f, round(c, 3)) for f, c in coef[:3]])

save_result("house-price", {
    "cfg": {"model": "Linear regression", "dataset": f"California Housing, {len(df):,} districts",
            "features": f"{len(FEATURES)} (3 engineered)", "split": "80 / 20", "optimizer": "least squares + gradient descent"},
    "metric": {"label": "Test R²", "value": f"{r2:.3f}"},
    "metrics": [{"label": "Test RMSE", "value": f"${rmse / 1000:,.1f}k"},
                {"label": "Mean-only baseline", "value": f"${baseline_rmse / 1000:,.1f}k"}],
    "curve": {"xLabel": "gradient descent step", "yLabel": "MSE ($100k²)", "x": list(range(1, 41)),
              "train": r(train_mse), "val": r(val_mse), "trainLabel": "train MSE", "valLabel": "test MSE"},
    "findings": [
        f"Median income is the strongest single signal (correlation {corr['median_income']:.2f} with price).",
        f"{capped:.1%} of districts were capped at $500k in the census, so I removed them instead of letting the model learn a fake ceiling.",
        f"Location matters: the biggest coefficients are {coef[0][0].replace('_', ' ')} and {coef[1][0].replace('_', ' ')}. Before cleaning, districts west of longitude −121.5° had a median value of ${west / 1000:,.0f}k vs ${east / 1000:,.0f}k further east.",
        f"My hand-written gradient descent reaches a test error of ${gd_rmse / 1000:,.1f}k after 40 steps, close to the exact least-squares solution (${rmse / 1000:,.1f}k).",
        f"The model explains {r2:.0%} of the variation in price and cuts the error from ${baseline_rmse / 1000:,.0f}k (always guessing the average) to ${rmse / 1000:,.0f}k. A straight line can't capture neighbourhood effects, which is the obvious next step.",
    ],
})
