package BackpropagationClassifier
import java.lang.Math
import java.util.Random

class Multilayer(noIns:Int, noHidden:Int, noOutputs:Int) {
  // sigmoid function - using tanh
  val sigmoid = (x:Double) => Math.tanh(x)
  val sigmoidPrimeOfY = (y:Double) => 1.0 - Math.pow(y,2)

  // set up the network
  val noInputs = noIns + 1 // bias is the +1 :)

  // arrays of activations
  val inputActivations = new Array[Double](noInputs)
  val hiddenActivations = new Array[Double](noHidden)
  val outputActivations = new Array[Double](noOutputs)

  // initialize the weights
  val inputWeights = Array.ofDim[Double](noInputs, noHidden)
  val outputWeights = Array.ofDim[Double](noHidden, noOutputs)
  // init weights to random values
  // TODO: Adjust these and check performance
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
  val inputLastChanges = Array.ofDim[Double](noInputs, noHidden)
  val outputLastChanges = Array.ofDim[Double](noHidden, noOutputs)


  def update(inputs:Array[Double]):Array[Double] = {
    // work on the inputs
    for (x <- 0 until noInputs - 1) {
      inputActivations(x) = inputs(x)
    }

    // work on the hidden layers
    for (x <- 0 until noHidden) {
      var sum = 0.0
      for (y <- 0 until noInputs) {
        sum += inputActivations(y) * inputWeights(y)(x)
      }
      hiddenActivations(x) = sigmoid(sum)
    }

    // work on the output layers
    for (x <- 0 until noOutputs) {
      var sum = 0.0
      for (y <- 0 until noHidden) {
        sum += hiddenActivations(y) * outputWeights(y)(x)
      }
      outputActivations(x) = sigmoid(sum)
    }

    outputActivations clone
  }

  def backPropogationTrain(targets:Array[Double], learningRate:Double, momentumRate:Double) = {
    // find error deltas first
    var error:Double = 0.0
    var change:Double = 0.0

    val outputDeltas = new Array[Double](noOutputs)
    for (x <- 0 until noOutputs) {
      error = targets(x) - outputActivations(x)
      outputDeltas(x) = sigmoidPrimeOfY(outputActivations(x)) * error
    }

    val hiddenDeltas = new Array[Double](noHidden)
    for (x <- 0 until noHidden) {
      error = 0.0
      for (y <- 0 until noOutputs) {
        error += outputDeltas(y) * outputWeights(x)(y)
      }
      hiddenDeltas(x) = sigmoidPrimeOfY(hiddenActivations(x)) * error
    }

    // update output and input weights
    for (x <- 0 until noHidden) {
      for (y <- 0 until noOutputs) {
        change = outputDeltas(y) * hiddenActivations(x)
        outputWeights(x)(y) = outputWeights(x)(y) + learningRate * change + momentumRate * outputLastChanges(x)(y)
        outputLastChanges(x)(y) = change
      }
    }

    for (x <- 0 until noInputs) {
      for (y <- 0 until noHidden) {
        change = hiddenDeltas(y) * inputActivations(x)
        inputWeights(x)(y) = inputWeights(x)(y) + learningRate * change + momentumRate * inputLastChanges(x)(y)
        inputLastChanges(x)(y) = change
      }
    }

    error = 0.0
    for (x <- 0 until targets.length) {
      error += 0.5 * Math.pow(targets(x)-outputActivations(x),2)
    }
    error
  }

  def test(patterns:Array[(Array[Double],Array[Double])]) = {
    for (x <- patterns) {
      val result = update(x._1)
      println("*******")
      for (y <- x._2) {
        print(y)
        print(" ")
      }
      println(":")
      for (y <- result) {
        println(y)
      }
      println("-----")
    }
  }

  def train(patterns:Array[(Array[Double],Array[Double])], iterations:Int, learningRate:Double, momentumRate:Double, testPatters:Array[(Array[Double],Array[Double])]) = {
    for (x <- 0 until iterations) {
      var error:Double = 0.0
      for (p <- patterns) {
        update(p._1) // update inputs
        error += backPropogationTrain(p._2, learningRate, momentumRate)
      }
      println("training error,%f" format error)
      error = 0.0
      for (p <- testPatters) {
        val result = update(p._1)
        for (x <- 0 until p._2.length) {
          error += 0.5 * Math.pow(p._2(x)-result(x),2)
        }
      }
      println("testing error,%f" format error)
    }
  }
}
