import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}

/**
  * Created by Juan on 12/11/2016.
  */
object MainLR {

  private final val SVM_TEXT = "svm"
  private final val LR_TEXT = "lreg"

  def main(args: Array[String]): Unit = {
    //FileUtils.cleanDirectory(new File("C:/Users/JuanManuel/git/LearningRate/target/tmp/"))

    val conf = new SparkConf().setAppName("LearningRate").setMaster("local")
    val sc = new SparkContext(conf)

    //val filePath = "inFiles/kddb-raw-libsvm"
    //val filePath = "inFiles/covtype.txt"
    //val filePath = "inFiles/sample_libsvm_data.txt"
    //val filePath = "/share/hadoop/spark-2.0.2-bin-hadoop2.7/lra/covtype.txt"
    //val filePath = "inFiles/covtype.txt"
    val filePath = args(0)
    val classMethod = args(1)
    val lrType = args(2)
    val nIterations = args(3).toInt

    classMethod match {
      case SVM_TEXT => SVMImpl.runSVM(sc, filePath, lrType, nIterations)
      case LR_TEXT => LogisticRegressionImpl.runLogisticRegression(sc, filePath, lrType, nIterations)
    }
  }
}