import json
import time
import flaml
from flaml import tune
from flaml.model import XGBoostEstimator
from pyspark.sql import SparkSession


spark = SparkSession.getActiveSession()
sc = spark.sparkContext
spark_executor_instances = int(spark.conf.get("spark.executor.instances"))

logManager = sc._jvm.org.apache.log4j.LogManager
logManager.getLogger("ml.dmlc").setLevel(sc._jvm.org.apache.log4j.Level.INFO)
logManager.getLogger("flaml").setLevel(sc._jvm.org.apache.log4j.Level.INFO)
log = logManager.getLogger("INFO")

train_df = spark.read.parquet("data/train.parquet/").withColumnRenamed(
    "is_fraud", "label"
)
valid_df = spark.read.parquet("data/valid.parquet/").withColumnRenamed(
    "is_fraud", "label"
)

XGBoosterEvaluator = sc._jvm.model_eval.XGBoostClassifierEval
HP_METRIC = "aucpr"
MODE = "max"
num_samples = 10
time_budget_s = 1800


def xgb_eval(config: dict) -> dict:
    fit_start_time = time.time()
    params = XGBoostEstimator(**config).params
    aucpr = XGBoosterEvaluator.fitPredictEval(
        json.dumps(params), train_df._jdf, valid_df._jdf, spark_executor_instances
    )
    time_to_fit = time.time() - fit_start_time
    # tune.report(_metric=HP_METRIC, metric_name=HP_METRIC, metric_mode=MODE, metric_value=aucpr, time_to_fit=time_to_fit, params=params)
    return {HP_METRIC: aucpr}


# load a built-in search space from flaml
flaml_xgb_search_space = XGBoostEstimator.search_space(
    (train_df.count(), len(train_df.columns))
)
# specify the search space as a dict from hp name to domain; you can define your own search space same way
config_search_space = {
    hp: space["domain"] for hp, space in flaml_xgb_search_space.items()
}
# give guidance about hp values corresponding to low training cost, i.e., {"n_estimators": 4, "num_leaves": 4}
# low_cost_partial_config = {
#     hp: space["low_cost_init_value"]
#     for hp, space in flaml_xgb_search_space.items()
#     if "low_cost_init_value" in space
# }
low_cost_partial_config = {
    "n_estimators": 5,
    "max_leaves": 5,
    "max_depth": 6,
    "subsample": 1.0,
    "colsample_bylevel": 1.0,
    "colsample_bytree": 1.0,
}

# initial points to evaluate
# points_to_evaluate = [
#     {
#         hp: space["init_value"]
#         for hp, space in flaml_xgb_search_space.items()
#         if "init_value" in space
#     }
# ]
# points_to_evaluate = [low_cost_partial_config]

# ray.init(spark_executor_instances=spark_executor_instances)
start_time = time.time()
analysis = tune.run(
    xgb_eval,
    metric=HP_METRIC,
    mode=MODE,
    config=config_search_space,
    low_cost_partial_config=low_cost_partial_config,
    # points_to_evaluate=points_to_evaluate,
    # metric_constraints=["precision", ">=", 0.9]
    use_ray=False,
    local_dir="logs/",
    time_budget_s=time_budget_s,
    num_samples=num_samples,
)

log.info(f"Search took {time.time() - start_time} seconds")
best_trial = analysis.get_best_trial(HP_METRIC, MODE, "all")
metric = best_trial.metric_analysis[HP_METRIC][MODE]
