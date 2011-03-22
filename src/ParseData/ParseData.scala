package ParseData

/**
 * The file runs and tests the ParseData Application
 */

object ParseData extends Application {
  override def main(args: Array[String]) {
    val dataLocation = "./data/"

    val r = new UrlDataReader(dataLocation)

    val part_1 = 0
    val part_2 = 1

    // Partition day 0 into 2 parts
    r.partitionData(0, 2)
    //val day = r.getDay(0)
    for (x <- r.getPartition(0, part_1)) {
      println(x.classification.toString)
    }
    //println(r.getDay(0).getItem(10000).classification.toString)
    //println(r.getDay(0).getItem(3).features.toString)
  }
}