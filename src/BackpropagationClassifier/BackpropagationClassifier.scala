package BackpropagationClassifier
import ParseData._
import java.lang.Math._

object BackpropagationClassifier extends Application {
  override def main(args: Array[String]) {
    val dataLocation = "./data/"
    val r = new UrlDataReader(dataLocation)

    // data parts
    val part_1 = 0
    val part_2 = 1

    // Partition day 0 into 2 parts
    r.partitionData(0, 2)
    val trainingData = r.getPartition(0, 0).toArray
    val testData = r.getPartition(0, 1).toArray

    // select some training data
    val trainingPatterns = new Array[(Array[Double], Array[Double])](trainingData.length)
    for (x <- 0 until trainingPatterns.length) {
      trainingPatterns(x) = (r.getArray(trainingData(x).features), List(trainingData(x).classification.toDouble).toArray)
    }
    val testPatterns = new Array[(Array[Double], Array[Double])](testData.length)
    for (x <- 0 until testPatterns.length) {
      testPatterns(x) = (r.getArray(testData(x).features), List(testData(x).classification.toDouble).toArray)
    }

    // make a new network
    val noFeatures = r.features.length
    //val noHidden = CalcNumberHiddenNodes.logBaseTwo(noFeatures)
    //val noHidden = CalcNumberHiddenNodes.novelApproach(trainingPatterns.length, noFeatures)
    val noHidden = 37
    val noOut = 1 // binary classification
    val net = new Multilayer(noFeatures, noHidden, noOut)

    // train the net on the training data
    val iterations = 1000
    val learningRate = 0.5
    val momentum = 0.01
    net.train(trainingPatterns, iterations, learningRate, momentum, testPatterns)
  }
}