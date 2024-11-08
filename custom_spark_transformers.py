# Databricks notebook source
# MAGIC %md
# MAGIC ## Databricks SparkML Quickstart: Model Training
# MAGIC This notebook focuses on creating custom transformers in a spark pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries
# MAGIC Import necessary libraries preinstalled on Databricks Runtime for Machine Learning.

# COMMAND ----------

# MAGIC %pip install sparkml_base_classes mlflow

# COMMAND ----------
# test
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer, StringIndexer
from sparkml_base_classes import TransformerBaseClass, EstimatorBaseClass
from pyspark.ml.regression import GBTRegressor
# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Read the dataset
# MAGIC
# MAGIC You can download the dataset from here: [Kaggle Link](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/). Please consult the following [source](https://archive.ics.uci.edu/dataset/186/wine+quality) from [Cortez et al., 2009]. Place the downloaded `.csv` fiile into the `./data/` directory

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import lit
df = spark.read.table("samples.nyctaxi.trips")
print('taxi',df.count())
# Concatenate the dataframes
data_df = spark.read.table("samples.wine_quality.wine")
print('wine',data_df.count())

# Make the quality into categories
data_df = data_df.withColumn("quality", F.when(data_df["quality"] > 7, "High").otherwise("Low"))

display(data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Split the dataset

# COMMAND ----------

train, test = data_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

df = train
df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Build the model
# MAGIC We want to build a ml model that predicts the 'alcohol' content in the wine. In partidcular, we would like to train a pipeline that performs three custom transformations on the dataset:
# MAGIC
# MAGIC 1. Custome transformer introduction: 

# COMMAND ----------

from sparkml_base_classes import TransformerBaseClass, EstimatorBaseClass
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType

# Custom Transformer 1: CustomImputer
class CustomImputer(TransformerBaseClass):
    """
    A custom Transformer that imputes missing values in a DataFrame.
    - Fills categorical columns with 'none'
    - Fills numeric columns with -99    """
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

    color_encoder = StringIndexer(inputCol="color", outputCol="indexed_color")
    
    addition_transformer = CustomAdder(inputCols=['fixed acidity',
                    'volatile acidity'], outputCol='total acidity')

    quality_target_encoder = TargetEncoder(inputCol='quality', targetCol='alcohol',outputCol='encoded_quality')

    vector_assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features")

    # Machine Learning model
    model = GBTRegressor(features_col="features", label_col="alcohol")

    stages=[color_encoder, addition_transformer, quality_target_encoder, vector_assembler, model]
    # Pipeline
    pipeline = Pipeline(stages=stages)
    return pipeline


# COMMAND ----------

import mlflow
mlflow.end_run()
pipeline = get_wine_data_model_pipeline()
pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)
display(predictions)

# COMMAND ----------

pipeline_model.write().overwrite().save("model_file")


# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.evaluation import RegressionEvaluator
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
    mlflow.log_metrics("root_mean_squared_error", rmse)

    # Log the model
    mlflow.spark.log_model(
        pipeline_model, artifact_path="custom_pipeline_model/")


# COMMAND ----------

