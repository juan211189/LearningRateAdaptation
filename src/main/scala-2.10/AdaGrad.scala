import org.apache.spark.mllib.optimization.Updater
//import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
//import breeze.linalg.{Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

@DeveloperApi
final class AdaGrad extends Updater {
  // Records the past values of the square values of the gradient
  private[this] var gradientsHistory: Array[Double] = _

  override def compute(
    weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double
  ): (Vector, Double) = {
    // Update the gradient history
    +=(gradient)
    // Convert old weights into a Breeze dense vector
    val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
    // Zip the learning rates with the old weights
    val sumSquareDerivative = learningRates.zip(brzWeights.toArray)
    // Calculate new weights
    val newWeights: Array[Double] = sumSquareDerivative.map{
      case (coef, weight) => weight * (1.0 -regParam * coef)}
    // Perform the matrices computation
    brzAxpy(-1.0, asBreeze(gradient.toArray), asBreeze(newWeights))
    // Computes the norm
    val norm = brzNorm(brzWeights, 2.0)
    // Return weights and loss
    (fromBreeze(asBreeze(newWeights)), 0.5 * regParam * norm * norm)
  }

  // Method used to update the gradient history
  private def +=(gradient: Vector): Unit = {
    val grad = gradient.toArray
    grad.view.zipWithIndex.foreach {
      case (g, index) => {
        if (gradientsHistory == null)
          gradientsHistory = Array.fill(grad.length)(0.0)
        val existingGradient = gradientsHistory(index)
        gradientsHistory.update(index, existingGradient + g*g)
      }
    }
  }

  // Compute the array of the new learning rates
  def learningRates = gradientsHistory.map(1.0/Math.sqrt(_))

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