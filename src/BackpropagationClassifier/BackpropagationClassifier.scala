package BackpropagationClassifier
import ParseData._

object BackpropagationClassifier extends Application {
  def multiLayerBackPropogationWithMomentum = {
    val trainingPatterns = new Array[(Array[Double], Array[Double])](trainingData.length)
    for (x <- 0 until trainingPatterns.length) {
      trainingPatterns(x) = (trainingData(x), outputs(x % outputs.length))
    }
    val testPatterns = new Array[(Array[Double], Array[Double])](testData.length)
    for (x <- 0 until trainingPatterns.length) {
      testPatterns(x) = (testData(x), outputs(x % outputs.length))
    }

    val net = new MultiLayer(noInputs, 14, noOutputs) // log(2)*7 ~= 3
    net.train(trainingPatterns, 1000, 0.1, 0.0, testPatterns) // no momentum
    net.test(testPatterns)
  }

  override def main(args: Array[String]) {
    val dataLocation = "./data/"
    val r = new UrlDataReader(dataLocation)

    // data parts
    val part_1 = 0
    val part_2 = 1

    // Partition day 0 into 2 parts
    r.partitionData(0, 2)

    // select some training data
    for(x <- )

    // make a new network
    val noFeatures = r.features.length
    val noHidden = ceil(log(noFeatures)/log(2)) // log base 2 of features
    val noOut = 1 // binary classification
    val net = new Multilayer(noFeatures, noHidden, noOut)


    // train the net on the training data
    //val day = r.getDay(0)
    for (x <- r.getPartition(0, part_1)) {
      println(x.classification.toString)
    }
    //println(r.getDay(0).getItem(10000).classification.toString)
    //println(r.getDay(0).getItem(3).features.toString)
  }
}