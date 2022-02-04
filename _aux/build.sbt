ThisBuild / version := "0.1"
ThisBuild / scalaVersion := "2.12.15"
ThisBuild / organization := "wise.com"
name := s"ModelEvaluator"
val sparkVersion = "3.2.0"
val hadoopVersion = "3.3.1"
libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "3.2.0" % "provided",
      "org.apache.spark" %% "spark-core" % "3.2.0" % "provided",
      "org.apache.spark" %% "spark-mllib" % "3.2.0" % "provided",
      "com.amazonaws" % "aws-java-sdk-bundle" % "1.12.129" % "provided",
      "org.apache.hadoop" % "hadoop-aws" % "3.3.1",
      "ml.dmlc" %% "xgboost4j" % "1.5.1",
      "ml.dmlc" %% "xgboost4j-spark" % "1.5.1"
  )
lazy val autoxgboostspark = (project in file("."))
  .settings(
    name := s"ModelEvaluator",
    assembly / mainClass := Some("model_eval.XGBoostEval"),
    assembly / assemblyJarName := s"xgb_model_eval-0.1_scala-${scalaVersion.value}_spark-${sparkVersion}_hadoop-${hadoopVersion}.jar"
  )

scalacOptions += "-deprecation"
