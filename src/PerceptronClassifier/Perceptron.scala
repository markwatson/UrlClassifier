package PerceptronClassifier
import java.lang.Math._

class Perceptron(noIns:Int, noOuts:Int) {
  val weights = Array.ofDim[Double](noIns, noOuts)

  def trainingRule(in:Double, out:Double, weight:Double) = {
    weight + in  * out
  }

  val activationFunction = (x:Double) => if(x < 0) -1.0 else 1.0

  def trainVector(vector: Array[Double], result:Array[Double]) = {
    for (x <- 0 until vector.length) {
      for (y <- 0 until result.length) {
        weights(x)(y) = trainingRule(vector(x), result(y), weights(x)(y))
      }
    }
  }

  def run(vector: Array[Double]) = {
    val output = new Array[Double](noOuts)

    for (y <- 0 until output.length) {
      for (x <- 0 until vector.length) {
        output(y) = output(y) + vector(x) * weights(x)(y)
      }
    }

    output
  }

  def train(patterns:Array[(Array[Double],Array[Double])], testPatterns:Array[(Array[Double],Array[Double])]) = {
    // train
    for (x <- patterns) {
      trainVector(x._1, x._2)
    }

    // test
    var amountWrong = 0
    for (x <- testPatterns) {
      // run it
      val out = run(x._1)
      var error: Double = 0.0
      for (y <- 0 until noOuts) {
        error += abs(out(y) - x._2(y))
        val output = activationFunction(out(y))
        println("real output v. correct output,%s,%s" format (output, x._2(y)))
        if (output != x._2(y)) {
          amountWrong += 1
        }
      }
      println("testing error,%f" format error)
    }
    println("amount wrong,%d" format amountWrong)
    println("total patterns,%d" format testPatterns.length)
  }
}