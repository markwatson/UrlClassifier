package BackpropagationClassifier
import java.lang.Math._
import java.util.Random

class MultilayerDynamic(noIns:Int, initNoHidden:Int, noOutputs:Int)
extends Multilayer(noIns, initNoHidden, noOutputs) {
  // This class is a multilayer that increases the number of hidden nodes to dynamically find the correct number of
  // nodes.
  // Based on research in this paper:
  // Y. Hirose, K. Yamashita, and S. Hijiya, "Back-propagation algorithm which varies the number of hidden units,"
  // Neural Networks, vol. 4, no. 1, pp. 61-66, 1991. [Online].
  // Available: http://dx.doi.org/10.1016/0893-6080(91)90032-Z

  // code for adding new hidden layers
  def addHiddenNode(relearn:Boolean = false) = {
    // relearn: means relearn all nodes each time
    noHidden += 1
    println("number hidden nodes,%d" format noHidden)

    // save current results
    val oldHiddenActivations = hiddenActivations
    val oldInputWeights = inputWeights
    val oldOutputWeights = outputWeights
    val oldInputLastChanges = inputLastChanges
    val oldOutputLastChanges = outputLastChanges

    // add new node
    hiddenActivations = new Array[Double](noHidden)
    inputWeights = Array.ofDim[Double](noInputs, noHidden)
    outputWeights = Array.ofDim[Double](noHidden, noOutputs)
    inputLastChanges = Array.ofDim[Double](noInputs, noHidden)
    outputLastChanges = Array.ofDim[Double](noHidden, noOutputs)

    // copy old results
    Array.copy(oldHiddenActivations, 0, hiddenActivations, 0, oldHiddenActivations.length)
    if (!relearn) {
      // input
      for (x <- 0 until noInputs) {
        for (y <- 0 until (noHidden-1)) {
          inputWeights(x)(y) = oldInputWeights(x)(y)
        }
      }
      // output
      for (x <- 0 until (noHidden-1)) {
        for (y <- 0 until noOutputs) {
          outputWeights(x)(y) = oldOutputWeights(x)(y)
        }
      }
      // last change buffers
      for (x <- 0 until noInputs) {
        for (y <- 0 until (noHidden-1)) {
          inputLastChanges(x)(y) = oldInputLastChanges(x)(y)
        }
      }
      for (x <- 0 until (noHidden-1)) {
        for (y <- 0 until noOutputs) {
          outputLastChanges(x)(y) = oldOutputLastChanges(x)(y)
        }
      }
    } else {
      val rnd = new Random
      val rndInRange = (x:Double, y: Double) => (y-x) * rnd.nextDouble + x
      for (x <- 0 until noInputs) {
        for (y <- 0 until noHidden) {
          inputWeights(x)(y) = rndInRange(0.0,0.2)
        }
      }
      for (x <- 0 until noHidden) {
        for (y <- 0 until noOutputs) {
          outputWeights(x)(y) = rndInRange(0.0,0.5)
        }
      }
      // momentum
      var inputLastChanges = Array.ofDim[Double](noInputs, noHidden)
      var outputLastChanges = Array.ofDim[Double](noHidden, noOutputs)
    }
  }

  def train(patterns:Array[(Array[Double],Array[Double])], iterations:Int, learningRate:Double,
                     momentumRate:Double, testPatters:Array[(Array[Double],Array[Double])],
                     errorThreshold:Double, addEvery:Int, relearn:Boolean, addOnChange:Double) = {
    // for each iteration
    var i:Int = 0
    var trainingError:Double = 0.0
    var testError:Double = 0.0
    var savedError:Double = 0.0
    do {
      i += 1

      // train on training data
      trainingError = 0
      for (p <- patterns) {
        update(p._1) // update inputs
        trainingError += backPropogationTrain(p._2, learningRate, momentumRate)
      }
      println("training error,%f" format trainingError)

      // possibly add a node

      if (i == 1) {
        savedError = trainingError
      }
      if ((i % addEvery) == 0) {
        if ((trainingError-savedError) < addOnChange) {//TODO: take out magic number
          addHiddenNode(relearn)
        }
        savedError = trainingError
      }

      // test on test data
      testError = 0
      for (p <- testPatters) {
        val result = update(p._1)
        for (x <- 0 until p._2.length) {
          testError += 0.5 * pow(p._2(x)-result(x),2)
        }
      }
      println("testing error,%f" format testError)

    } while (i < iterations || trainingError > errorThreshold)
  }
}

