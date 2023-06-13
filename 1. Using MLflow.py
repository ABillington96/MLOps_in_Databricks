# Databricks notebook source
# MAGIC %md 
# MAGIC # MLflow Demo Notebook
# MAGIC This notebook will demonstrate how to use MLflow in databricks to train a machine learning model, log the parameters & metrics, register a model, before finally preparing the model for production.
# MAGIC ## Helper Functions
# MAGIC Just a few helper functions to format get the data ready for model training! 

# COMMAND ----------

from pyspark.sql import *
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta
import mlflow.pyfunc


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).timestamp())


rounded_unix_timestamp_udf = udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    """
    Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features respectively.
    """
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_pickup_datetime"], lit(15)),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_dropoff_datetime"], lit(30)),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df

# COMMAND ----------

# MAGIC %md 
# MAGIC # Load the dataset
# MAGIC Load the NYC taxi dataset which will be used for training a regression model to predict taxi fares.

# COMMAND ----------

# Load the raw NYC taxi dataset from the selection default databricks datasets. 
raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
# Use helper function to round the data to 15 and 30 minute intervals
taxi_data = rounded_taxi_data(raw_data)

# Display the dataset
display(taxi_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the data for training
# MAGIC The most barebones steps are taken to prepare the data for training a model, splitting into a training and test set.
# MAGIC
# MAGIC We will look more at what can be done with the data in the next section on feature store.

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Get the feature and label names
features_and_label = taxi_data.columns

# Collect data into a Pandas array for training
data = taxi_data.toPandas()[features_and_label]

# Split the data into train and test sets
train, test = train_test_split(data, random_state=42)

# Separate the features by dropping the label column
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)

# Separate the labels by selecting only the relevant column
y_train = train.fare_amount
y_test = test.fare_amount

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train an SKLearn Random Forest Regressor
# MAGIC Here we will train a Random Forest regressor, logging the parameters and the model performance metrics to MLflow.

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Start mlflow autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Define the parameters for the random forest
    n_estimators = 100
    max_depth = 6
    max_features = 3
  
    # Create and train a random forest regresor using the predefined parameters
    rf_model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf_model.fit(X_train, y_train)
  
    # Get predictions for the test data
    rf_predictions = rf_model.predict(X_test)

    # Calculate performance metrics
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_mse = mean_squared_error(y_test, rf_predictions)
 
    # Log the performance metrics to mlflow
    mlflow.log_metric("Mean absolute error", rf_mae)
    mlflow.log_metric("Neab squared error", rf_mse)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a Light GBM
# MAGIC Here we will train a second model, a Light GBM, logging the parameters and the model performance metrics to MLflow. We can then easily compare the performance of the two models to one another.

# COMMAND ----------

import lightgbm as lgb
import mlflow.lightgbm

# Start mlflow autologging
mlflow.lightgbm.autolog()

# Prepare training dataset for use in the gbm
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)

with mlflow.start_run():
    # Define parameters for the gbm
    lgb_param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
    lgb_num_rounds = 100

    # Train a gbm using the predefined parameters
    lgb_model = lgb.train(lgb_param, train_lgb_dataset, lgb_num_rounds)

    # Get predictions for the test data
    lgb_predictions = lgb_model.predict(X_test)

    # Calculate performance metrics
    lgb_mae = mean_absolute_error(y_test, lgb_predictions)
    lgb_mse = mean_squared_error(y_test, lgb_predictions)

    # Log the performance metrics to ML
    mlflow.log_metric("Mean absolute error", lgb_mae)
    mlflow.log_metric("Neab squared error", lgb_mse)

# COMMAND ----------


