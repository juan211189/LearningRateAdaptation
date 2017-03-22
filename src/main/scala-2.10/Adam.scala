import org.apache.spark.mllib.optimization.Updater
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

import scala.math._


final class Adam extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (mean, variance): (BV[Double], BV[Double]) = (null, null)
        
        val epsilon : Float = 1e-8f
        val decayFactor : Float = 1 - epsilon
        val beta1 : Float = 0.9f
        val beta2 : Float = 0.999f
        
        var time: Int = 0
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
        
                if( mean == null) {
                        mean            = BDV.zeros[Double](gradientSize)
                        variance        = BDV.zeros[Double](gradientSize)
                }
                
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquared: BV[Double] = brzGradients :+ brzGradients
                
                time = time + 1
        
                val currentIterStepSize : Double = ( stepSize * math.sqrt(1.0f - math.pow(beta2, time)) / ( 1.0f - math.pow(beta1,time)))
                
                val currentBeta1 : Double       = beta1 * math.pow(decayFactor, time - 1)
                
                val currentMeanLefthandside  : BV[Double]   = mean :* currentBeta1
                val currentMeanRighthandside : BV[Double]   = brzGradients :* (1.0f - currentBeta1)
                val currentMean              : BV[Double]   = currentMeanLefthandside + currentMeanRighthandside
                
                val currentVarianceLefthandside  : BV[Double] = variance :* beta2.toDouble
                val currentVarianceRighthandside : BV[Double] = brzGradientsSquared :* ( 1.0f - beta2).toDouble
                val currentVariance              : BV[Double] = currentVarianceLefthandside + currentVarianceRighthandside
                val currentVarianceSquareroot    : BV[Double] = currentVariance.copy.map(item => math.sqrt(item))
                
                val currentStepLefthandside      : BV[Double] = currentMean :+ currentIterStepSize
                val currentStepRighthandside     : BV[Double] = currentVarianceSquareroot + epsilon.toDouble
                val currentStep                  : BV[Double] = currentStepLefthandside :/ currentStepRighthandside
        
        
                mean            = currentMean
                variance        = currentVariance
        
        
                brzWeights -= currentStep
                
                (fromBreeze(brzWeights), 0)
        }
        
        def asBreeze(values: Array[Double]): BV[Double] = new BDV[Double](values)
        
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
        
}

final class AdamL1 extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (mean, variance): (BV[Double], BV[Double]) = (null, null)
        
        val epsilon : Float = 1e-8f
        val decayFactor : Float = 1 - epsilon
        val beta1 : Float = 0.9f
        val beta2 : Float = 0.999f
        
        var time: Int = 0
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
        
                if( mean == null) {
                        mean            = BDV.zeros[Double](gradientSize)
                        variance        = BDV.zeros[Double](gradientSize)
                }
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquared: BV[Double] = brzGradients :+ brzGradients
        
                time = time + 1
        
                val currentIterStepSize : Double = ( stepSize * math.sqrt(1.0f - math.pow(beta2, time)) / ( 1.0f - math.pow(beta1,time)))
        
                val currentBeta1 : Double       = beta1 * math.pow(decayFactor, time - 1)
        
                val currentMeanLefthandside  : BV[Double]   = mean :* currentBeta1.toDouble
                val currentMeanRighthandside : BV[Double]   = brzGradients :* (1.0f - currentBeta1)
                val currentMean              : BV[Double]   = currentMeanLefthandside + currentMeanRighthandside
        
                val currentVarianceLefthandside  : BV[Double] = variance :* beta2.toDouble
                val currentVarianceRighthandside : BV[Double] = brzGradientsSquared :* ( 1.0f - beta2).toDouble
                val currentVariance              : BV[Double] = currentVarianceLefthandside + currentVarianceRighthandside
                val currentVarianceSquareroot    : BV[Double] = currentVariance.copy.map(item => math.sqrt(item))
        
                val currentStepLefthandside      : BV[Double] = currentMean :+ currentIterStepSize
                val currentStepRighthandside     : BV[Double] = currentVarianceSquareroot + epsilon.toDouble
                val currentStep                  : BV[Double] = currentStepLefthandside :/ currentStepRighthandside
        
        
                mean            = currentMean
                variance        = currentVariance
        
                brzWeights -= currentStep
                
                val shrinkageVal = regParam * currentIterStepSize
                var i = 0
                val len = brzWeights.length
                while (i < len) {
                        val wi = brzWeights(i)
                        brzWeights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
                        i += 1
                }
                
                
                (fromBreeze(brzWeights), brzNorm(brzWeights, 1.0) * regParam)
        }
        
        def asBreeze(values: Array[Double]): BV[Double] = new BDV[Double](values)
        
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
        
}

final class AdamL2 extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (mean, variance): (BV[Double], BV[Double]) = (null, null)
        
        val epsilon : Float = 1e-8f
        val decayFactor : Float = 1 - epsilon
        val beta1 : Float = 0.9f
        val beta2 : Float = 0.999f
        
        var time: Int = 0
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
        
                if( mean == null) {
                        mean            = BDV.zeros[Double](gradientSize)
                        variance        = BDV.zeros[Double](gradientSize)
                }
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquared: BV[Double] = brzGradients :+ brzGradients
        
                time = time + 1
        
                val currentIterStepSize : Double = ( stepSize * math.sqrt(1.0f - math.pow(beta2, time)) / ( 1.0f - math.pow(beta1,time)))
        
                val currentBeta1 : Double       = beta1 * math.pow(decayFactor, time - 1)
        
                val currentMeanLefthandside  : BV[Double]   = mean :* currentBeta1.toDouble
                val currentMeanRighthandside : BV[Double]   = brzGradients :* (1.0f - currentBeta1)
                val currentMean              : BV[Double]   = currentMeanLefthandside + currentMeanRighthandside
        
                val currentVarianceLefthandside  : BV[Double] = variance :* beta2.toDouble
                val currentVarianceRighthandside : BV[Double] = brzGradientsSquared :* ( 1.0f - beta2).toDouble
                val currentVariance              : BV[Double] = currentVarianceLefthandside + currentVarianceRighthandside
                val currentVarianceSquareroot    : BV[Double] = currentVariance.copy.map(item => math.sqrt(item))
        
                val currentStepLefthandside      : BV[Double] = currentMean :+ currentIterStepSize
                val currentStepRighthandside     : BV[Double] = currentVarianceSquareroot + epsilon.toDouble
                val currentStep                  : BV[Double] = currentStepLefthandside :/ currentStepRighthandside
        
        
                mean            = currentMean
                variance        = currentVariance
                brzWeights :*= (1.0 - currentIterStepSize * regParam)
        
                brzWeights -= currentStep
                
                val norm = brzNorm(brzWeights, 2.0)
                
                (fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
        }
        
        
        def asBreeze(values: Array[Double]): BV[Double] = new BDV[Double](values)
        
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
        
}