import org.apache.spark.mllib.optimization.Updater
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm => brzNorm}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

import scala.math._

final class AdaDelta extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (accumulatorGradient, accumulatorDelta): (BV[Double], BV[Double]) = (null, null)
        
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
                
                if( accumulatorGradient == null) {
                        accumulatorGradient = BDV.zeros[Double](gradientSize)
                        accumulatorDelta    = BDV.zeros[Double](gradientSize)
                }
                
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
                
                val brzGradientsSquare: BV[Double] = brzGradients.copy
                brzGradientsSquare :*= brzGradients
                
                val leftHandSide: BV[Double] = accumulatorGradient :* rho.toDouble
                val rightHandSide: BV[Double] = brzGradientsSquare :* (1.0f - rho).toDouble
        
                accumulatorGradient =  leftHandSide + rightHandSide
                
                
                val currentDeltaNumerator       : BV[Double] = accumulatorDelta + epsilon.toDouble
                val currentDeltaDenominator     : BV[Double] = brzGradients :* currentDeltaNumerator
                
                currentDeltaNumerator.map( item => math.sqrt(item))
                currentDeltaDenominator.map( item => math.sqrt(item))
                
                val currentDelta : BV[Double] = currentDeltaNumerator :/ currentDeltaDenominator
                
                
                val currentDeltaSquared: BV[Double] = currentDelta :* currentDelta
                val deltaLeftHandSide: BV[Double] =  accumulatorDelta :* rho.toDouble
                val deltaRightHandSide: BV[Double] = currentDeltaSquared :* (1.0f - rho).toDouble
        
                accumulatorDelta =   deltaLeftHandSide + deltaRightHandSide
                
                print(iter)
        
                brzWeights -= accumulatorDelta
                
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

final class AdaDeltaL1 extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (accumulatorGradient, accumulatorDelta): (BV[Double], BV[Double]) = (null, null)
        
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
        
                if( accumulatorGradient == null) {
                        accumulatorGradient = BDV.zeros[Double](gradientSize)
                        accumulatorDelta    = BDV.zeros[Double](gradientSize)
                }
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquare: BV[Double] = brzGradients.copy
                brzGradientsSquare :*= brzGradients
        
                val leftHandSide: BV[Double] = accumulatorGradient :* rho.toDouble
                val rightHandSide: BV[Double] = brzGradientsSquare :* (1.0f - rho).toDouble
        
                val newAccumulatorGradient : BV[Double] =  leftHandSide + rightHandSide
        
                accumulatorGradient = newAccumulatorGradient
        
        
                val currentDeltaNumerator       : BV[Double] = accumulatorDelta + epsilon.toDouble
                val currentDeltaDenominator     : BV[Double] = brzGradients :* currentDeltaNumerator
        
                currentDeltaNumerator.map( item => math.sqrt(item))
                currentDeltaDenominator.map( item => math.sqrt(item))
        
                val currentDelta : BV[Double] = currentDeltaNumerator :/ currentDeltaDenominator
        
        
                val currentDeltaSquared: BV[Double] = currentDelta :* currentDelta
                val deltaLeftHandSide: BV[Double] =  accumulatorDelta :* rho.toDouble
                val deltaRightHandSide: BV[Double] = currentDeltaSquared :* (1.0f - rho).toDouble
        
                val newDelta : BV[Double] =   deltaLeftHandSide + deltaRightHandSide
        
                accumulatorDelta = newDelta
        
                print(iter)
        
                brzWeights -= accumulatorDelta
                
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

final class AdaDeltaL2 extends Updater {
        
        // Records the past values of the square values of the gradient
        
        var (accumulatorGradient, accumulatorDelta): (BV[Double], BV[Double]) = (null, null)
        
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
        
                if( accumulatorGradient == null) {
                        accumulatorGradient = BDV.zeros[Double](gradientSize)
                        accumulatorDelta    = BDV.zeros[Double](gradientSize)
                }
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                val brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                val brzGradientsSquare: BV[Double] = brzGradients.copy
                brzGradientsSquare :*= brzGradients
        
                val leftHandSide: BV[Double] = accumulatorGradient :* rho.toDouble
                val rightHandSide: BV[Double] = brzGradientsSquare :* (1.0f - rho).toDouble
        
                val newAccumulatorGradient : BV[Double] =  leftHandSide + rightHandSide
        
                accumulatorGradient = newAccumulatorGradient
        
        
                val currentDeltaNumerator       : BV[Double] = accumulatorDelta + epsilon.toDouble
                val currentDeltaDenominator     : BV[Double] = brzGradients :* currentDeltaNumerator
        
                currentDeltaNumerator.map( item => math.sqrt(item))
                currentDeltaDenominator.map( item => math.sqrt(item))
        
                val currentDelta : BV[Double] = currentDeltaNumerator :/ currentDeltaDenominator
        
        
                val currentDeltaSquared: BV[Double] = currentDelta :* currentDelta
                val deltaLeftHandSide: BV[Double] =  accumulatorDelta :* rho.toDouble
                val deltaRightHandSide: BV[Double] = currentDeltaSquared :* (1.0f - rho).toDouble
        
                val newDelta : BV[Double] =   deltaLeftHandSide + deltaRightHandSide
        
                accumulatorDelta = newDelta
        
                print(iter)
                
                brzWeights :*= (1.0 - stepSize * regParam)
        
                brzWeights -= accumulatorDelta
                
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