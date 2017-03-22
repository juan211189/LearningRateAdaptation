import org.apache.spark.{SparkConf, SparkContext}

object MainLR {
        
        private final val SVM_TEXT = "svm"
  
        private final val LR_TEXT = "lreg"

  
        def main(args: Array[String]): Unit = {
                
                val conf = new SparkConf().setAppName("LearningRate").setMaster("local")
    
                val sc = new SparkContext(conf)
    
                val filePath = args(0)
    
                val classMethod = args(1)
    
                val lrType = args(2)
    
                val nIterations = args(3).toInt

                val regType = args(4)
                
                val regParam = args(5).toDouble
    
                val stepSize = args(6).toDouble
                
                
                
                classMethod match {
      
                        case SVM_TEXT => SVMImpl.runSVM(sc, filePath, lrType, nIterations, regType, regParam, stepSize)
      
                        case LR_TEXT => LogisticRegressionImpl.runLogisticRegression(sc, filePath, lrType, nIterations, regType, regParam, stepSize)
    
                }
  
        }
}