package ParseData

import scala.io.Source
import scala.collection.mutable
import scala.util.Random

class UrlDataReader(dataRoot: String) {
  // these are constants for the data structure
  private val featuresFile = dataRoot + "FeatureTypes"
  private def dayFile(day: Int) = dataRoot + "Day" + day.toString + ".svm"
  private val partitions = new mutable.HashMap[(Int, Int), Partition];

  class Partition(day: Int, partitionNumber: Int, numberPartitions: Int)
  extends Iterable[Int] {
    val svmFile = new Svm(dayFile(day), features)
    val items = new Array[Int](svmFile.numSamples / numberPartitions)
    val iterator = items.iterator
    val rnd = new Random()

    for (i  <- 0.until(items.length)) {
      //x = svmFile.getItem(rnd.nextInt(svmFile.numSamples / numberPartitions))
      items(i) = rnd.nextInt(items.length)
    }
  }

  // data structures
  val features = (for (x <- Source.fromFile(featuresFile).getLines)
                  yield x.toInt).toList

  def getDay(day: Int) = {
    new Svm(dayFile(day), features)
  }

  def partitionData(day: Int, numberPartitions: Int) = {
    for (i <- 0.until(numberPartitions)) {
      val np = new Partition(i, day, numberPartitions)
      partitions.put((day, i), np)
    }
  }

  def getPartition(day: Int, partition: Int) = {
    val dayData = getDay(day)
    dayData.getItem(0)

    for (x <- partitions((day, partition))) yield dayData.getItem(x)
  }
}