/**
SP-BRAIN 1.0
@author Valerio Morfino
@version 1.0 

BrainScala Model
*/

package brain.scala

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.linalg.SparseVector
import scala.util.control.Breaks._


object BrainScalaModel{
  //Load a persisted model from file
  def loadModel(filePath:String):BrainScalaModel={
    val sparksession = SparkSession.builder().appName("Spark-Brain").getOrCreate()
      var df = sparksession.read.csv(filePath+".metadata")
      val version = df.collect()(0)(0).asInstanceOf[String]
      val funzTxt = df.collect()(1)(0).asInstanceOf[String]
      val trainingTime = df.collect()(2)(0).asInstanceOf[String].toDouble
      val ptot = df.collect()(3)(0).asInstanceOf[String].toLong
      val ntot = df.collect()(4)(0).asInstanceOf[String].toLong
      val funz = BrainUtil.fromDFOfArrayStringToArrayOfArrayDouble(sparksession.read.csv(filePath+".model"))
      return new BrainScalaModel(version, funz,funzTxt,trainingTime,ptot,ntot)
  }
}

class BrainScalaModel (val version: String, val funz: Array[Array[Double]], val funzTxt: String, val trainingTime: Double, val trainingPos: Long, val trainingNeg: Long){

  def saveModel(filePath:String)={
    val sc = SparkSession.builder().getOrCreate().sparkContext
    sc.parallelize(funz.map(_.mkString(",")),1).saveAsTextFile(filePath+".model")
    sc.parallelize(Array(version, funzTxt,trainingTime,trainingPos,trainingNeg),1).saveAsTextFile(filePath+".metadata")
  }

  def transform(instances: DataFrame):DataFrame = {
    val classifySequence = udf(BrainTransformer.checkSeq(_: SparseVector, _: String ): Double)
    val funzSerializable=funz.map(_.mkString(";")).mkString(",")
    return instances.withColumn("class",classifySequence(col("features"),lit(funzSerializable)))
  }

}

object BrainTransformer{
  def checkSeq(seq: SparseVector, funzStr: String ): Double ={
    val funz = funzStr.split(",").map(_.split(";")).map(_.map(_.toDouble))

    val fnIterator = funz.iterator
    while (fnIterator.hasNext)
    {
      val term = fnIterator.next()
      var termFlag=true
      breakable{
        for (i <-term.indices){
            if ((seq(i)==1.0 && term(i) == 2.0) || (seq(i)==0.0 && term(i)==1.0)) {
              termFlag= false
              break;
              }
        }
      }
      if (termFlag) return 1.0

    }
    return 0.0
  }
  /*
  def checkSeq(seq: SparseVector, funzStr:Seq[Seq[Double]]): Double ={
    val funz = str.split(",").map(_.split(";")).map(_.map(_.toDouble))

    val fnIterator = funz.iterator
    while (fnIterator.hasNext)
    {
      val term = fnIterator.next()
        for (i <-term.indices){
            if ((seq(i)==1.0 && term(i) == 2.0) || (seq(i)==0.0 && term(i)==1.0)) {
              return 0.0
              }
        }
    }
    return 1.0
  }
  */
}
