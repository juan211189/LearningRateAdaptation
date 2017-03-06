import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.Updater
//import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
//import breeze.linalg.{Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

@DeveloperApi
final class ConstantRate() extends Updater {

  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    //val thisIterStepSize = stepSize / math.sqrt(iter)
    val thisIterStepSize = 0.5
    val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    brzAxpy(-thisIterStepSize, asBreeze(gradient.toArray), brzWeights)
    val norm = brzNorm(brzWeights, 2.0)

    (fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
  }

  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  def asBreeze(values: Array[Double]): BV[Double] = new BDV[Double](values)
}