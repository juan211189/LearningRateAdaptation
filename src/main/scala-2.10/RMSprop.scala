import org.apache.spark.mllib.optimization.Updater
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

import scala.math._


final class RMSprop extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (accumulatorN, accumulatorG, accumulatorDelta): (BV[Double], BV[Double], BV[Double]) = (null, null, null)
        
        val epsilon : Float = 1e-8f
        
        val rho : Float = 0.9f
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
                
                if( accumulatorN == null) {
                        accumulatorN            = BDV.zeros[Double](gradientSize)
                        accumulatorG            = BDV.zeros[Double](gradientSize)
                        accumulatorDelta        = BDV.zeros[Double](gradientSize)
                }
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquare: BV[Double] = brzGradients.copy
                brzGradientsSquare :*= brzGradients
                
                
                val accumulatorNLefthandside: BV[Double]        = brzGradientsSquare :* (1.0f - rho).toDouble
                val accumulatorNRighthandside: BV[Double]       = accumulatorN :* rho.toDouble
                
                accumulatorN = accumulatorNLefthandside + accumulatorNRighthandside
                
                
                val accumulatorGLefthandside: BV[Double]        = brzGradients :* (1.0f - rho).toDouble
                val accumulatorGRighthandside: BV[Double]       = accumulatorG :* rho.toDouble
                
                accumulatorG = accumulatorGLefthandside + accumulatorGRighthandside
        
                
                val accumulatorGSquared: BV[Double]             = accumulatorDelta :* accumulatorDelta
                var intermediateDenominator: BV[Double]         = accumulatorN - accumulatorGSquared
                intermediateDenominator :+ epsilon.toDouble
                intermediateDenominator.map(item => math.sqrt(item))
                intermediateDenominator = brzGradients :/ intermediateDenominator
                
                
                val accumulatorDeltaLefthandside = accumulatorDelta :* rho.toDouble
                val accumulatorDeltaRighthandside = intermediateDenominator :* stepSize
                
                accumulatorDelta = accumulatorDeltaLefthandside - accumulatorDeltaRighthandside
                        
                brzWeights += accumulatorDelta
                
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

final class RMSpropL1 extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (accumulatorN, accumulatorG, accumulatorDelta): (BV[Double], BV[Double], BV[Double]) = (null, null, null)
        
        val epsilon : Float = 1e-8f
        
        val rho : Float = 0.9f
        
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
                if( accumulatorN == null) {
                        accumulatorN            = BDV.zeros[Double](gradientSize)
                        accumulatorG            = BDV.zeros[Double](gradientSize)
                        accumulatorDelta        = BDV.zeros[Double](gradientSize)
                }
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquare: BV[Double] = brzGradients.copy
                brzGradientsSquare :*= brzGradients
        
        
                val accumulatorNLefthandside: BV[Double]        = brzGradientsSquare :* (1.0f - rho).toDouble
                val accumulatorNRighthandside: BV[Double]       = accumulatorN :* rho.toDouble
        
                accumulatorN = accumulatorNLefthandside + accumulatorNRighthandside
        
        
                val accumulatorGLefthandside: BV[Double]        = brzGradients :* (1.0f - rho).toDouble
                val accumulatorGRighthandside: BV[Double]       = accumulatorG :* rho.toDouble
        
                accumulatorG = accumulatorGLefthandside + accumulatorGRighthandside
        
        
                val accumulatorGSquared: BV[Double]             = accumulatorDelta :* accumulatorDelta
                var intermediateDenominator: BV[Double]         = accumulatorN - accumulatorGSquared
                intermediateDenominator :+ epsilon.toDouble
                intermediateDenominator.map(item => math.sqrt(item))
                intermediateDenominator = brzGradients :/ intermediateDenominator
        
        
                val accumulatorDeltaLefthandside = accumulatorDelta :* rho.toDouble
                val accumulatorDeltaRighthandside = intermediateDenominator :* stepSize
        
                accumulatorDelta = accumulatorDeltaLefthandside - accumulatorDeltaRighthandside
        
                brzWeights += accumulatorDelta
                
                
                val shrinkageVal = regParam * stepSize
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

final class RMSpropL2 extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (accumulatorN, accumulatorG, accumulatorDelta): (BV[Double], BV[Double], BV[Double]) = (null, null, null)
        
        val epsilon : Float = 1e-8f
        
        val rho : Float = 0.9f
        
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
                if( accumulatorN == null) {
                        accumulatorN            = BDV.zeros[Double](gradientSize)
                        accumulatorG            = BDV.zeros[Double](gradientSize)
                        accumulatorDelta        = BDV.zeros[Double](gradientSize)
                }
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquare: BV[Double] = brzGradients.copy
                brzGradientsSquare :*= brzGradients
        
        
                val accumulatorNLefthandside: BV[Double]        = brzGradientsSquare :* (1.0f - rho).toDouble
                val accumulatorNRighthandside: BV[Double]       = accumulatorN :* rho.toDouble
        
                accumulatorN = accumulatorNLefthandside + accumulatorNRighthandside
        
        
                val accumulatorGLefthandside: BV[Double]        = brzGradients :* (1.0f - rho).toDouble
                val accumulatorGRighthandside: BV[Double]       = accumulatorG :* rho.toDouble
        
                accumulatorG = accumulatorGLefthandside + accumulatorGRighthandside
        
        
                val accumulatorGSquared: BV[Double]             = accumulatorDelta :* accumulatorDelta
                var intermediateDenominator: BV[Double]         = accumulatorN - accumulatorGSquared
                intermediateDenominator :+ epsilon.toDouble
                intermediateDenominator.map(item => math.sqrt(item))
                intermediateDenominator = brzGradients :/ intermediateDenominator
        
        
                val accumulatorDeltaLefthandside = accumulatorDelta :* rho.toDouble
                val accumulatorDeltaRighthandside = intermediateDenominator :* stepSize
        
                accumulatorDelta = accumulatorDeltaLefthandside - accumulatorDeltaRighthandside
        
                brzWeights :*= (1.0 - stepSize * regParam)
        
                brzWeights += accumulatorDelta
        
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
        

