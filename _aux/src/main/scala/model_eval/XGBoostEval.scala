package model_eval

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, XGBoostClassifier}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object XGBoostClassifierEval {
	def fitPredictEval(xgbParamsJsonString: String, train_df: Dataset[Row], valid_df: Dataset[Row], cores: Integer = 1): Double = {
		implicit val formats = org.json4s.DefaultFormats
		val xgbParams = parse(xgbParamsJsonString).extract[Map[String, Any]]
		val booster = new XGBoostClassifier(xgbParams)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setProbabilityCol("probability")
      .setEvalSets(Map("eval_test" -> train_df))
      .setMissing(0.0f)
      .setNumWorkers(cores)
      .setCheckpointInterval(10)
      .setCheckpointPath("/tmp/xgboost_checkpoints/")
      .setUseExternalMemory(false)
      .setTimeoutRequestWorkers(6000L)

		booster.set(booster.trackerConf, TrackerConf(60 * 60 * 1000, "scala"))
		booster.set(booster.killSparkContextOnWorkerFailure, false)
		val classifier = booster.fit(train_df)
		val result = classifier.transform(valid_df)
		val scoreAndLabels = result.select(
      col("prediction").cast("double").alias("prediction"),
      col("label").cast("double").alias("prediction")
    ).rdd.map(
			row => (row(0).asInstanceOf[Double], row(1).asInstanceOf[Double])
		)
		val aucpr = new BinaryClassificationMetrics(scoreAndLabels).areaUnderPR()
		return aucpr
	}

  // def fit
  // def predict
  // def eval
}
