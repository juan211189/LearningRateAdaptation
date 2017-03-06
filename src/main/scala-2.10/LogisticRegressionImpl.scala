import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

/**
  * Created by Juan on 12/11/2016.
  */
object LogisticRegressionImpl {
  def runLogisticRegression(sc: SparkContext, filePath: String, lrType: String, nIterations: Int): Unit = {
    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, filePath)

    // Split data into training (70%) and test (30%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    /*val lRegP = null

    lrType match {
      case "adagrad" => print("Adagrad updater")
        val lRegP = new LogisticRegressionWithSGD().optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
      //.setUpdater(new AdaGrad())

      case "adadelta" => print("Adadelta updater")
        lRegP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
      //.setUpdater(new AdaGrad())
      case "adam" => print("Adam updater")
        lRegP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
      //.setUpdater(new AdaGrad())
      case "rmsprop" => print("RMSProp updater")
        lRegP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
      //.setUpdater(new AdaGrad())
      case "constant" => print("Constant rate updater")
        lRegP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
          .setUpdater(new ConstantRate)
      case "default" => print("Default updater")
        lRegP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
    }*/

    val numIterations = 10000
    val lRegP = new LogisticRegressionWithLBFGS()
    lRegP.optimizer
      .setNumIterations(numIterations)
      .setRegParam(0.1)
      //.setUpdater(new AdaGrad(0))

    // // Generate Logistic Regression model and measure execution time
    val _startTime: Long = System.currentTimeMillis
    val model = lRegP.run(training)
    val _endTime = System.currentTimeMillis


    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    println(s"Accuracy = $accuracy")

    // Processing time
    println("Processing time " + ((_endTime - _startTime) / 1000) + " seconds.")
    println("Processing time " + ((_endTime - _startTime)) + " milliseconds.")

    /*
    // Save and load model
    model.save(sc, "target/tmp/scalaLogisticRegressionWithLBFGSModel")
    val sameModel = LogisticRegressionModel.load(sc,
      "target/tmp/scalaLogisticRegressionWithLBFGSModel")
    */
    sc.stop()
  }
}
