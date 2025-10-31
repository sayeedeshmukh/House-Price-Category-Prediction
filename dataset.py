from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.Series(data.target, name="MedHouseVal")
print(df)