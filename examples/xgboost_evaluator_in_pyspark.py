import json
from pyspark.sql import SparkSession

spark = SparkSession.getActiveSession()
sc = spark.sparkContext
spark_executor_instances = int(spark.conf.getAll["spark.executor.instances"] | 2)

logManager = sc._jvm.org.apache.log4j.LogManager
logManager.getLogger("ml.dmlc").setLevel("INFO")

train_df = spark.read.parquet("data/train.parquet/").withColumnRenamed(
    "is_fraud", "label"
)
valid_df = spark.read.parquet("data/valid.parquet/").withColumnRenamed(
    "is_fraud", "label"
)

XGBoosterEvaluator = sc._jvm.model_eval.XGBoostClassifierEval

xgbParamsDict = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "aucpr",
    "maximize_evaluation_metrics": True,
    "num_round": 10,
    "num_early_stopping_rounds": 5,
    "verbosity": 2,
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "single_precision_histogram": False,
    "max_bins": 256,
    "use_external_memory": False,
    "num_classes": 2,
    "n_estimators": 252,
    "eta": 0.05,
    "max_depth": 10,
    "colsample_by_tree": 0.8,
    "colsample_by_level": 0.8,
    "min_child_weight": 5,
    "subsample": 0.6,
    "verbosity": 2,
}

xgbParamsJsonString = json.dumps(xgbParamsDict)

aucpr = XGBoosterEvaluator.fitPredictEval(
    xgbParamsJsonString, train_df._jdf, valid_df._jdf, spark_executor_instances
)
