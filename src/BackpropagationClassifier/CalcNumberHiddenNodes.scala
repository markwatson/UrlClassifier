package BackpropagationClassifier
import java.lang.Math._

/**
 * Calculates number of optimal hidden nodes for a BNN.
 *
 * N: Number of training pairs
 * d: input dimension
 *
 */

object CalcNumberHiddenNodes {
  def logBaseTwo(d: Int) = {
    ceil(log(d)/log(2)).toInt
  }

  def novelApproach(N: Int, d: Int) = {
    // calculates number of hidden nodes based on the novel approach:
    // S. Xu and L. Che, "A novel approach for determining the optimal number of hidden layer neurons for FNN's and
    // its application in data mining," in 5th International Conference on Information Technology and Applications
    // (ICITA 2008), 2008.
    ceil(pow(N/(d*log(N)),0.5)).toInt
  }
}