import org.apache.spark.mllib.optimization.Updater
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

import scala.math._

final class AdaGrad extends Updater {
        
        // Records the past values of the square values of the gradient
        var historicalGradients : BV[Double] = null
        val epsilon : Float = 1e-8f
        
        override def compute(
                weightsOld: Vector,
                gradient: Vector,
                stepSize: Double,
                iter: Int,
                regParam: Double
                ): (Vector, Double) = {
                
                val gradientSize = gradient.size
        
                val thisIterStepSize:Double = stepSize
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                var brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                var accumulatorSquare: BV[Double] = brzGradients.copy
                
        
                print(iter)
                
                if(historicalGradients == null){
                        historicalGradients = BDV.zeros[Double](gradientSize)
                }
        
                accumulatorSquare :*= brzGradients
                
                val accumulatorSquareCopy: BV[Double] = accumulatorSquare.copy
        
                accumulatorSquare :+= historicalGradients
                
                accumulatorSquare :+= epsilon.toDouble
                
                historicalGradients :+= accumulatorSquareCopy
                
                accumulatorSquare.map( item => math.sqrt(item))
                
                brzGradients :/= accumulatorSquare
        
                brzAxpy(-thisIterStepSize, brzGradients, brzWeights)
        
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

final class AdaGradL1 extends Updater {
        
        // Records the past values of the square values of the gradient
        var historicalGradients : BV[Double] = null
        val epsilon : Float = 1e-8f
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
                val thisIterStepSize:Double = stepSize
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                var brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                var accumulatorSquare: BV[Double] = brzGradients.copy
        
        
                print(iter)
        
                if(historicalGradients == null){
                        historicalGradients = BDV.zeros[Double](gradientSize)
                }
        
                accumulatorSquare :*= brzGradients
        
                val accumulatorSquareCopy: BV[Double] = accumulatorSquare.copy
        
                accumulatorSquare :+= historicalGradients
        
                accumulatorSquare :+= epsilon.toDouble
        
                historicalGradients :+= accumulatorSquareCopy
        
                accumulatorSquare.map( item => math.sqrt(item))
        
                brzGradients :/= accumulatorSquare
        
                brzAxpy(-thisIterStepSize, brzGradients, brzWeights)
        
        
                val shrinkageVal = regParam * thisIterStepSize
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

final class AdaGradL2 extends Updater {
        
        // Records the past values of the square values of the gradient
        var historicalGradients : BV[Double] = null
        val epsilon : Float = 1e-7f
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
        
                val gradientSize = gradient.size
        
                val thisIterStepSize:Double = stepSize
        
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                var brzGradients: BV[Double] = asBreeze(gradient.toArray).toDenseVector
        
                var accumulatorSquare: BV[Double] = brzGradients.copy
        
        
                print(iter)
        
                if(historicalGradients == null){
                        historicalGradients = BDV.zeros[Double](gradientSize)
                }
        
                accumulatorSquare :*= brzGradients
        
                val accumulatorSquareCopy: BV[Double] = accumulatorSquare.copy
        
                accumulatorSquare :+= historicalGradients
        
                accumulatorSquare :+= epsilon.toDouble
        
                historicalGradients :+= accumulatorSquareCopy
        
                accumulatorSquare.map( item => math.sqrt(item))
        
                brzGradients :/= accumulatorSquare
                
                brzWeights :*= (1.0 - thisIterStepSize * regParam)
                
                brzAxpy(-thisIterStepSize, brzGradients, brzWeights)
                
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


