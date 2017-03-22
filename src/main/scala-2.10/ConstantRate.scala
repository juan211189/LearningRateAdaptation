import org.apache.spark.mllib.optimization.Updater
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

import scala.math._


final class ConstantRate extends Updater {
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
                
                val thisIterStepSize:Double = 0.5f
                
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                
                brzAxpy(-thisIterStepSize, asBreeze(gradient.toArray), brzWeights)
                
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

final class ConstantRateL1 extends Updater {
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
                
                val gradientSize = gradient.size
                
                val thisIterStepSize:Double = 0.5f
                
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                
                brzAxpy(-thisIterStepSize, asBreeze(gradient.toArray), brzWeights)
                
                
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

final class ConstantRateL2 extends Updater {
        
        override def compute(
                                    weightsOld: Vector,
                                    gradient: Vector,
                                    stepSize: Double,
                                    iter: Int,
                                    regParam: Double
                            ): (Vector, Double) = {
                
                val gradientSize = gradient.size
                
                val thisIterStepSize:Double = 0.5f
                
                val brzWeights: BV[Double] = asBreeze(weightsOld.toArray).toDenseVector
                
                brzWeights :*= (1.0 - thisIterStepSize * regParam)
                brzAxpy(-thisIterStepSize, asBreeze(gradient.toArray), brzWeights)
                
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