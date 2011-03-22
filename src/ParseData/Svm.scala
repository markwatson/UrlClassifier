package ParseData

import scala.io.Source
import scala.collection.immutable.IntMap

class Svm(file: String, features: List[Int]) {
  // load the data into a raw data
  private val rawData = loadFile

  private def parseFeature(in: String) = {
    val parts = in.split(":")
    (parts.head.toInt, parts.last.toFloat)
  }

  private def isFeature(in: (Int, Float)) = features.exists(x => in._1 == x)

  def loadFile = {
    val ret = Array.newBuilder[List[String]]

    for (x <- Source.fromFile(file).getLines) {
      ret += x.split(" ").toList
    }

    ret.result
  }

  // get item returns this
  class Item(classificationIn: Int, featuresIn:IntMap[Float]) {
    val classification = classificationIn
    val features = featuresIn
    var used = false
  }
  def getItem(index: Int) = {
    val data = rawData(index)

    // can only be 1 or -1
    var test_result = 0
    if (data.head(0) != '+') {
      test_result = -1
    } else {
      test_result = 1
    }

    new Item(test_result,
           IntMap[Float](data.tail.map(parseFeature).filter(isFeature): _*))
  }

  def numSamples = {
    rawData.length
  }
}