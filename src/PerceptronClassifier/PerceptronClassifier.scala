package PerceptronClassifier
import ParseData._
import java.lang.Math._

object PerceptronClassifier extends Application {
  override def main(args: Array[String]) {
    val dataLocation = "./data/"

    // data parts
    val part_1 = 0
    val part_2 = 1


    // make a new network
    val tmpR = new UrlDataReader(dataLocation)
    val noFeatures = tmpR.features.length
    val noOut = 1 // binary classification
    val net = new Perceptron(noFeatures, noOut)

    val days = 121
    for (day <- 0 until days) {
      println("DAY %d DATA" format day)

      // Partition day 0 into 2 parts
      var r = new UrlDataReader(dataLocation)
      r.partitionData(day, 2)
      val trainingData = r.getPartition(day, 0).toArray
      val testData = r.getPartition(day, 1).toArray

      // select some training data
      val trainingPatterns = new Array[(Array[Double], Array[Double])](trainingData.length)
      for (x <- 0 until trainingPatterns.length) {
        trainingPatterns(x) = (r.getArray(trainingData(x).features), List(trainingData(x).classification.toDouble).toArray)
      }
      val testPatterns = new Array[(Array[Double], Array[Double])](testData.length)
      for (x <- 0 until testPatterns.length) {
        testPatterns(x) = (r.getArray(testData(x).features), List(testData(x).classification.toDouble).toArray)
      }

      // train the net on the training data
      val learningRate = 0.3
      net.train(trainingPatterns, testPatterns)
      r = null
    }
  }
}