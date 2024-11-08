# Databricks notebook source
# MAGIC %md
# MAGIC # Custom Spark Transformers: From Simple to Advanced
# MAGIC 
# MAGIC This tutorial demonstrates how to create custom transformers in Spark ML pipelines and save complete fitted pipelines for production use. We'll explore:
# MAGIC 
# MAGIC 1. **Basic Transformer**: A simple missing value imputer
# MAGIC 2. **Intermediate Transformer**: A feature combiner that adds columns
# MAGIC 3. **Advanced Transformer**: A target encoder that requires both fit and transform steps
# MAGIC 
# MAGIC Key learning objectives:
# MAGIC - Building custom transformers with increasing complexity
# MAGIC - Creating end-to-end ML pipelines with custom transformations
# MAGIC - Saving fitted pipelines for production deployment
# MAGIC - Using MLflow to track and version your models
# MAGIC 
# MAGIC We'll use a wine quality dataset to predict alcohol content, showing how custom transformers can enhance your ML pipeline and be deployed to production.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC First, let's import our dependencies and load the wine quality dataset.

# COMMAND ----------

# MAGIC %pip install sparkml_base_classes mlflow xgboost

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from sparkml_base_classes import TransformerBaseClass, EstimatorBaseClass
from xgboost.spark import SparkXGBRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Read the dataset
# MAGIC
# MAGIC We'll load the Wine Quality Dataset directly from the UCI Machine Learning Repository:
# MAGIC - Red wine data: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
# MAGIC - White wine data: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
# MAGIC 
# MAGIC Reference: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# COMMAND ----------

# Load and prepare wine quality datasets
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Load data and add color labels

# Read the CSV files directly into pandas DataFrames
red_wine_pd = pd.read_csv(red_wine_url, sep=';')
white_wine_pd = pd.read_csv(white_wine_url, sep=';')

# Convert pandas DataFrames to Spark DataFrames
red_wine = spark.createDataFrame(red_wine_pd)
white_wine = spark.createDataFrame(white_wine_pd)

# Add a 'color' column to each DataFrame
red_wine = red_wine.withColumn("color", F.lit("red"))
white_wine = white_wine.withColumn("color", F.lit("white"))

# Concatenate the dataframes
data_df = white_wine.union(red_wine)

# Convert quality scores to binary categories (High/Low)
data_df = data_df.withColumn("quality", F.when(data_df["quality"] > 7, "High").otherwise("Low"))

# Display sample of prepared dataset
display(data_df.limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Split the dataset

# COMMAND ----------

# Split data into training (80%) and test (20%) sets
train, test = data_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training set size: {train.count()}")
print(f"Test set size: {test.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Building the ML Pipeline
# MAGIC 
# MAGIC Our goal is to predict wine alcohol content using custom transformers in a Spark ML pipeline. We'll demonstrate three types of transformers:
# MAGIC 
# MAGIC 1. **CustomImputer**: Handles missing values in both numeric and categorical columns
# MAGIC 2. **CustomAdder**: Combines acidity features into a new meaningful feature
# MAGIC 3. **TargetEncoder**: Encodes categorical quality ratings using mean alcohol content
# MAGIC 
# MAGIC ### Why Custom Transformers?
# MAGIC - Some sklearn-like functionality isn't available in Spark (e.g., target encoding)
# MAGIC - Complex feature engineering often requires custom logic
# MAGIC - Custom transformers can be saved and reused in production pipelines

# COMMAND ----------

from sparkml_base_classes import TransformerBaseClass, EstimatorBaseClass
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer

# Custom Transformer 1: CustomImputer
class CustomImputer(TransformerBaseClass):
    """
    Example 1: Basic Transformer
    
    A simple custom transformer that handles missing values:
    - Replaces nulls in string columns with 'none'
    - Replaces nulls in numeric columns with -99
    
    This demonstrates the basics of custom transformation without fitting.
    """
    @keyword_only
    def __init__(self):
        super().__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        # Identify string and non-string columns
        string_cols = [c[0] for c in df.dtypes if c[1] == 'string']
        non_string_cols = [c[0] for c in df.dtypes if c[1] != 'string']

        # Fill string columns with 'none' and non-string columns with -99
        df = df.fillna('none', subset=string_cols)
        df = df.fillna(-99, subset=non_string_cols)
        return df

# Custom Transformer 2: CustomAdder    
class CustomAdder(TransformerBaseClass):
    """
    Example 2: Intermediate Transformer
    
    Combines multiple numeric columns through addition.
    Shows how to work with multiple input columns and create derived features.
    """
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super().__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.withColumn(self._outputCol, sum(F.col(c) for c in self._inputCols))
        return df

 # Custom Transformer 3: TargetEncoder       
class TargetEncoderModel(TransformerBaseClass):
    @keyword_only
    def __init__(self, inputCol=None, targetCol="y", outputCol=None, target_map_json=None):
        super().__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        target_map = spark.read.json(spark.sparkContext.parallelize([self._target_map_json]))
        df = df.join(target_map, self._inputCol,'left')
        df = df.withColumn(self._outputCol, F.col("mean"))
        return df.drop("mean")

class TargetEncoder(EstimatorBaseClass):
    """
    Example 3: Advanced Transformer
    
    A target encoder that learns mean statistics during fit() and applies them during transform().
    Demonstrates the full estimator-transformer pattern for stateful transformations.
    """
    @keyword_only
    def __init__(self, inputCol=None, targetCol="y", outputCol=None):
        super().__init__()

    def _fit(self, df: DataFrame):
        assert self._inputCol in df.columns, f"Column {self._inputCol} not in dataframe. Available columns: {df.columns}"
        target_map = df.groupBy(self._inputCol).agg(F.mean(self._targetCol).alias("mean"))
        target_map_json = "[" + ",".join(target_map.toJSON().collect()) + "]"
        return TargetEncoderModel(inputCol=self._inputCol, targetCol=self._targetCol, outputCol=self._outputCol, target_map_json=target_map_json)

# COMMAND ----------

def get_wine_data_model_pipeline() -> Pipeline:
    """
    Builds an end-to-end pipeline combining our custom transformers:
    1. Encodes the wine color
    2. Combines acidity features
    3. Applies target encoding to quality
    4. Assembles features for the final XGBoost model
    """
    # Encode categorical color feature
    color_encoder = StringIndexer(inputCol="color", outputCol="indexed_color")
    
    addition_transformer = CustomAdder(inputCols=['fixed acidity',
                    'volatile acidity'], outputCol='total acidity')

    quality_target_encoder = TargetEncoder(inputCol='quality', targetCol='alcohol',outputCol='encoded_quality')

        # Assembling features, make sure you do not include the target column as feature column. This one is later utilised as labelCol parameter in the ML model
    feature_cols = [#'quality',
                    'fixed acidity',
                    'volatile acidity',
                    'citric acid',
                    'residual sugar',
                    'chlorides',
                    'free sulfur dioxide',
                    'total sulfur dioxide',
                    'density',
                    'pH',
                    'sulphates',
                    #'alcohol',
                    #'color',
                    'indexed_color',
                    'total acidity',
                    'encoded_quality']
    
    vector_assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features")

    # Machine Learning model
    model = SparkXGBRegressor(features_col="features", label_col="alcohol")

    stages=[color_encoder, addition_transformer, quality_target_encoder, vector_assembler, model]
    # Pipeline
    pipeline = Pipeline(stages=stages)
    return pipeline


# COMMAND ----------

# Train and evaluate the pipeline
import mlflow
mlflow.end_run()

# Create and fit the pipeline
pipeline = get_wine_data_model_pipeline()
pipeline_model = pipeline.fit(train)

# Make predictions on test set
predictions = pipeline_model.transform(test)

# Display sample predictions
display(predictions.select("alcohol", "prediction", "features").limit(5))

# COMMAND ----------

# Save the trained pipeline model
model_path = "wine_quality_pipeline"
pipeline_model.write().overwrite().save(model_path)
print(f"Model saved to: {model_path}")


# COMMAND ----------

# MLflow tracking setup
import mlflow
import mlflow.spark
from pyspark.ml.evaluation import RegressionEvaluator

# Configure MLflow tracking
mlflow.set_tracking_uri('databricks')
# Example: Fit and save the pipeline using MLFlow
with mlflow.start_run(run_name='Linear Regression Wine') as run:
    pipeline = get_wine_data_model_pipeline()
    pipeline_model = pipeline.fit(train)

    # Transforming the test data using the model
    predictions = pipeline_model.transform(test)

    # Evaluate the model to get actual metrics
    evaluator = RegressionEvaluator(
        labelCol="alcohol", predictionCol="prediction")

    # Log metrics
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})

    # Log model metrics
    mlflow.log_metric("root_mean_squared_error", rmse)

    # Log the model
    mlflow.spark.log_model(
        pipeline_model, artifact_path="custom_pipeline_model/")


# COMMAND ----------


