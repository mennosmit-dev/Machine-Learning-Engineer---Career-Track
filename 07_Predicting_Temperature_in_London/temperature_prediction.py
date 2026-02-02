#This script is used to predict temperature in London via OLS, Decision Tree, and Random Forest for a variety of hyperparamters, 
# Results are logged to MLFLOW.

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# SET MODE
mlflow.set_experiment("Weather_Prediction")

# Step 1: load data
df = pd.read_csv("london_weather.csv")
df.head()
df.info()

# Changing format of date column to work with it
df["date"] = pd.to_datetime(df["date"], format = "%Y%m%d")

# Now retrieving month and year and putting in dataframe
df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month

# Plotting temperature over time: first one shows clearly a trend (climate change),second clearly shows seasonality
#sns.lineplot(data=df, x=df["year"], y=df["mean_temp"])
#sns.lineplot(data=df, x=df["month"], y=df["mean_temp"])

# Insight into correlations (for feature selection) 
c_matrix = df.corr()
sns.heatmap(c_matrix)

#: With mean_temp as depedent, sunshine, global_radition,  max_temp, and min_temp show significant correlation;
#  month and year slight correlation, but I include both based on earlier graphs
regressor_columns = ["sunshine", "global_radiation", "max_temp", "min_temp", "year", "month"]

# Getting rid of observations with NA values in dependent variable
df = df.dropna(subset = ["mean_temp"])

# Splitting data in train and test
X_train, X_test, y_train, y_test = train_test_split(df[regressor_columns].values, df["mean_temp"].values, test_size=0.3)

# Now filling missed values and normalising 
imp = SimpleImputer()
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
norm = StandardScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)


for idx, depth in enumerate(range(1, 17, 2)):
    run_name = f"run_{idx}"
    with mlflow.start_run(run_name=run_name):
        # Make models
        ols = LinearRegression().fit(X_train, y_train)
        tree = DecisionTreeRegressor(max_depth = depth, random_state = 42).fit(X_train, y_train)
        forest = RandomForestRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)

        # Evaluate
        ols_rmse = mean_squared_error(y_test, ols.predict(X_test), squared = False)
        tree_rmse = mean_squared_error(y_test, tree.predict(X_test), squared = False)
        forest_rmse = mean_squared_error(y_test, forest.predict(X_test), squared = False)

        # Log it
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("rmse_ols", ols_rmse)
        mlflow.log_metric("rmse_tree", tree_rmse)
        mlflow.log_metric("rmse_forest", forest_rmse)

# Look at results
experiment_results = mlflow.search_runs()
experiment_results

