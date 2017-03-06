import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * Created by Juan on 12/11/2016.
  */
object SVMImpl {
  def runSVM(sc: SparkContext, filePath: String, lrType: String, nIterations: Int): Unit = {
    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, filePath)

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val svmP = new SVMWithSGD()

    lrType match {
      case "adagrad" => print("Adagrad updater")
        svmP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
          .setUpdater(new AdaGrad)
      case "adadelta" => print("Adadelta updater")
        svmP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
          //.setUpdater(new AdaGrad())
      case "adam" => print("Adam updater")
        svmP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
          //.setUpdater(new AdaGrad())
      case "rmsprop" => print("RMSProp updater")
        svmP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
          //.setUpdater(new AdaGrad())
      case "constant" => print("Constant rate updater")
        svmP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
          .setUpdater(new ConstantRate)
      case "default" => print("Default updater")
        svmP.optimizer
          .setNumIterations(nIterations)
          .setRegParam(0.1)
    }

    // Generate initial weights
    //val values = Array.fill(54)(0.5)
    //val initialWeights: Vector = Vectors.dense(values)

    // Generate SVM model and measure execution time
    val _startTime: Long = System.currentTimeMillis
    //val model = SVMWithSGD.train(training, numIterations)
    val model = svmP.run(training)
    val _endTime = System.currentTimeMillis

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)

    // AUROC
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)

    // Processing time
    println("Processing time " + ((_endTime - _startTime) / 1000) + " seconds.")
    println("Processing time " + ((_endTime - _startTime)) + " milliseconds.")

    /*
    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr
    println("PRC: " + PRC)

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc
    */

    // Save and load model
    //model.save(sc, "target/tmp/scalaSVMWithSGDModel")
    //val sameModel = SVMModel.load(sc, "target/tmp/scalaSVMWithSGDModel")

    sc.stop()
  }
}
