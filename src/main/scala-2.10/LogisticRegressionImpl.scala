import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils

object LogisticRegressionImpl {
        
        def runLogisticRegression(sc: SparkContext, filePath: String, lrType: String, nIterations: Int, regType: String, regPar: Double, stepSize: Double): Unit = {
        
                // Load training data in LIBSVM format.
    
                val data = MLUtils.loadLibSVMFile(sc, filePath)

    
                // Split data into training (70%) and test (30%).
    
                val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    
                val training = splits(0).cache()
    
                val test = splits(1)

    
                // Run training algorithm to build the model
    
                val lRegP = new LogisticRegressionWithSGD()

    
                lrType match {
      
                        case "adagrad" => print("Adagrad updater")
        
                                regType match {
        
                                        case "L0" => print("No Regularization")
                
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdaGrad)

                                        case "L1" => print("L1 Regularization")
        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdaGradL1)
        
                                        case "L2" => print("L2 Regularization")
        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdaGradL2)
        
                                }
                                
                        case "adadelta" => print("Adadelta updater")
        
                                regType match {
                
                                        case "L0" => print("No Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdaDelta)
                
                                        case "L1" => print("L1 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdaDeltaL1)
                
                                        case "L2" => print("L2 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdaDeltaL2)
                
                                }
      
                        case "adam" => print("Adam updater")
                                regType match {
                
                                        case "L0" => print("No Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new Adam)
                
                                        case "L1" => print("L1 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdamL1)
                
                                        case "L2" => print("L2 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new AdamL2)
                
                                }


                        case "rmsprop" => print("RMSProp updater")
                                regType match {
                
                                        case "L0" => print("No Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new RMSprop)
                
                                        case "L1" => print("L1 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new RMSpropL1)
                
                                        case "L2" => print("L2 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new RMSpropL2)
                
                                }


                        case "constant" => print("Constant rate updater")
        
                                regType match {
                
                                        case "L0" => print("No Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new ConstantRate)
                
                                        case "L1" => print("L1 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new ConstantRateL1)
                
                                        case "L2" => print("L2 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new ConstantRateL2)
                
                                }
      
                        case "default" => print("Default updater")
        
                                regType match {
                
                                        case "L0" => print("No Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new SimpleUpdater)
                
                                        case "L1" => print("L1 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new L1Updater)
                
                                        case "L2" => print("L2 Regularization")
                        
                                                lRegP.optimizer
                                                        .setNumIterations(nIterations)
                                                        .setRegParam(regPar)
                                                        .setStepSize(stepSize)
                                                        .setUpdater(new SquaredL2Updater)
                
                                }

                }
        
        
                // Generate LR model and measure execution time
        
                val _startTime: Long = System.currentTimeMillis
        
                val model = lRegP.run(training)
        
                val _endTime = System.currentTimeMillis
        
        
                println("Training time " + ((_endTime - _startTime) / 1000) + " seconds.")
        
                println("Training time " + ((_endTime - _startTime)) + " milliseconds.")
        
        
        
        
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
        
        
                sc.stop()
    
        }
}
