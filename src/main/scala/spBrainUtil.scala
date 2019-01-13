
package brain.scala

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.SparseVector
import scala.util.control.Breaks._

object BrainUtil {
//Create an array of Array of Double from a DataFrame containing SparseVector of Double
def fromDFOfSparseVectorToArrayOfArrayDouble(df: DataFrame): Array[Array[Double]]={
  var iter = df.takeAsList(df.count().asInstanceOf[Int]).iterator
  var arrOfarr = Array.empty[Array[Double]]
  while (iter.hasNext)
    arrOfarr:+= iter.next()(0).asInstanceOf[SparseVector].toArray
  return arrOfarr
}
//Create an array of Array of Double from a DataFrame containing SparseVector of String
def fromDFOfArrayStringToArrayOfArrayDouble(df: DataFrame): Array[Array[Double]]={
  var iter = df.takeAsList(df.count().asInstanceOf[Int]).iterator
  var arrOfarr = Array.empty[Array[Double]]
  while (iter.hasNext)
    arrOfarr:+= iter.next().toSeq.toArray.map(_.asInstanceOf[String].toDouble)
  return arrOfarr
}

//Filter positive Instaces satisfying the term of function
//The function works on Array[Array[Double]]
//Because of the presence of Dont'cares in the function, the convention used for term's coding is:
// * = 0.0
// 1 = 1.0
// 0 = 2.0
def filterPositiveInstances(instances: Array[Array[Double]], term: Array[Double]): Array[Array[Double]] ={
  var survivors = Array.empty[Array[Double]]

  var posIter = instances.iterator
  while (posIter.hasNext)
  {
    var instance = posIter.next()
    breakable{
      for (i <-term.indices){
          if ((instance(i)==1.0 && term(i) == 2.0) || (instance(i)==0.0 && term(i)==1.0)) {
            survivors:+= instance
            break
            }
      }
   }
  }
  return survivors
}

//Function used to filter positive Instaces satisfying the term of the inferred function.
//To function can be used in a spark SQL context after registration as UDF.
//This registration is maded in fit method
//Because of the presence of Dont'cares in the function, the convention used for term's coding is:
// * = 0.0
// 1 = 1.0
// 0 = 2.0
def checkPreservePositiveInstances(instance: SparseVector, term: Seq[Double]): Boolean ={
  for (i <- instance.toArray.indices)
    if ((instance(i)==1.0 && term(i)==2.0) || (instance(i)==0.0 && term(i)==1.0)) return true
    return false
}

//This function is used to check if an element of Sij set (positive or negative) have to be deleted
//The function work on SparseVector and can be used as UDF in a Spark query
//An element have to be deleted if:
//the bit at position top = 1 if top is related to a positive term, so the index is < n
//the bit at position top = 0 if top is related to a positive term, so the index is > n
//The function return true if instance have to be preserved (not deleted)
def checkPreserveSijElement(instance: SparseVector, top: Int, n: Int): Boolean = {
  var indiceTop=0
  var gamma=0.0
  if (top>n-1){
      indiceTop=top-n
      //gamma=false
  }else{
      indiceTop=top
      gamma=1.0
  }
  return (instance(indiceTop)==gamma)
}

}
