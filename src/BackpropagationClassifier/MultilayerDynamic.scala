package BackpropagationClassifier
import java.lang.Math._

class MultilayerDynamic(noIns:Int, noOutputs:Int)
extends Multilayer(noIns, 1, noOutputs) {
  // This class is a multilayer that increases the number of hidden nodes to dynamically find the correct number of
  // nodes.
  // Based on research in this paper:
  // Y. Hirose, K. Yamashita, and S. Hijiya, "Back-propagation algorithm which varies the number of hidden units,"
  // Neural Networks, vol. 4, no. 1, pp. 61-66, 1991. [Online].
  // Available: http://dx.doi.org/10.1016/0893-6080(91)90032-Z

  // code for adding new hidden layers
  def addHiddenNode = {
    noHidden += 1
    println("number hidden nodes,%d" format noHidden)

    // new list of hidden activations
    val oldHiddenActivations = hiddenActivations
    hiddenActivations = new Array[Double](noHidden)
    Array.copy(oldHiddenActivations, 0, hiddenActivations, 0, oldHiddenActivations.length)

    // add input weights (new ones initialized to 0)
    val oldInputWeights = inputWeights
    inputWeights = Array.ofDim[Double](noInputs, noHidden)
    for (x <- 0 until noInputs) {
      for (y <- 0 until (noHidden-1)) {
        inputWeights(x)(y) = oldInputWeights(x)(y)
      }
    }

    // add output weights (new ones initialized to 0)
    val oldOutputWeights = outputWeights
    outputWeights = Array.ofDim[Double](noHidden, noOutputs)
    for (x <- 0 until (noHidden-1)) {
      for (y <- 0 until noOutputs) {
        outputWeights(x)(y) = oldOutputWeights(x)(y)
      }
    }

    // modify size of last change buffers
    val oldInputLastChanges = inputLastChanges
    inputLastChanges = Array.ofDim[Double](noInputs, noHidden)
    for (x <- 0 until noInputs) {
      for (y <- 0 until (noHidden-1)) {
        inputLastChanges(x)(y) = oldInputLastChanges(x)(y)
      }
    }

    val oldOutputLastChanges = outputLastChanges
    outputLastChanges = Array.ofDim[Double](noHidden, noOutputs)
    for (x <- 0 until (noHidden-1)) {
      for (y <- 0 until noOutputs) {
        outputLastChanges(x)(y) = oldOutputLastChanges(x)(y)
      }
    }
  }

  def train(patterns:Array[(Array[Double],Array[Double])], iterations:Int, learningRate:Double,
                     momentumRate:Double, testPatters:Array[(Array[Double],Array[Double])],
                     errorThreshold:Double, addEvery:Int) = {
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
        if ((trainingError-savedError) < 1) {//TODO: take out magic number
          addHiddenNode
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

