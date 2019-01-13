/**
SP-BRAIN 1.0
@author Valerio Morfino
@version 1.0 - Cross Join

Relevance calculated using MapReduce
*/



package brain.scala
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.broadcast
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.broadcast
import scala.util.control.Breaks._

  object BrainScalaMRCROSS{

    def geTver():String = {
      return ("Brain Scala 1.0 - MRCROSS MapReduce + Cross Join Sij + UDF (legacy V modified)")
    }


    def maxRelevance(positive:SparseVector, negative:SparseVector, p: Long, q: Long, n: Int): SparseVector  = {
        var den=0.0

        //Counts differences between positive and negative instances
        for (k <- positive.toArray.indices){
            if (positive(k)!=negative(k)) den +=1
          }
        var v = Array.fill[Double] (n*2) (0.0)

        //Compute relevance
        var oneOnDen = 1/den.asInstanceOf[Double]
        for (k <- positive.toArray.indices){

          if (positive(k)==1.0 && negative(k)==0.0)
            v(k) +=  oneOnDen
          else if (positive(k)==0.0 && negative(k)==1.0)
            v(k+n) += oneOnDen
        }

        return (new org.apache.spark.ml.linalg.DenseVector(v)).toSparse
      }

    def maxRelevanceReduce(xr: org.apache.spark.sql.Row, yr: org.apache.spark.sql.Row,n: Int): org.apache.spark.sql.Row ={
      val x = xr(0).asInstanceOf[SparseVector]
      val y = yr(0).asInstanceOf[SparseVector]
      var v = Array.fill[Double] (n*2) (0.0)
      for (k <- x.toArray.indices) {
        v(k) = x(k) + y(k)
      }
      return Row((new org.apache.spark.ml.linalg.DenseVector(v)).toSparse)
    }

  //BRAIN Training
  //The method accept:
  //positiveTotPar as Dataframe containing SparseVector. This is the dataset containing positive Instances  (Cases with class=1.0 or True)
  //negativeTotPar as Dataframe containing SparseVector. This is the dataset containing positive Instances  (Cases with class=1.0 or True)
  //Verbose: Boolean. If true prints execution Log. Defaul = False
  //useCache: Use cache. Default = True
  //RETURN: BrainScalaMRBPModel, a model fitted on training Data.
  def fit(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = false, useCache: Boolean = true): BrainScalaModel = {
    if (useCache){
       negativeTotPar.cache()
       positiveTotPar.cache()
    }

	//Get the current Spark Context
	val sc = positiveTotPar.rdd.context

	val t0 = System.nanoTime()

	if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
	if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

	//Register the User Define Function checkPreserveSijElement
  val checkPreserveSijElementUDF = udf(BrainUtil.checkPreserveSijElement(_: SparseVector, _: Int, _: Int): Boolean)
  //Register UDF for check positive instances satisfied by function term
  val checkPreservePositiveInstancesUDF = udf(BrainUtil.checkPreservePositiveInstances(_: SparseVector, _:  Seq[Double] ): Boolean)
  //Register function for max relevance computing
  val maxRelevanceUDF = udf(maxRelevance(_: SparseVector, _: SparseVector, _: Long, _: Long, _: Int): SparseVector)


	var f1 = ""
	var funz1 = Array.empty[Array[Double]]

	//Count number of features
	val n= positiveTotPar.head()(0).asInstanceOf[SparseVector].size
	if (verbose)
    println ("n: " + n)

	//Count positive and negative instances
	var ptot=positiveTotPar.count()
	val ntot=negativeTotPar.count()

  //Add a unique ID to positive and negative instances
  var positiveTot = positiveTotPar.withColumn("pid",monotonically_increasing_id())
  val negativeTot = negativeTotPar.withColumnRenamed("features","neg_features").withColumn("nid",monotonically_increasing_id())

  //Compute the cross Join and calculate relevances of each couple of positive adn negative instances
  //In order to optimize the join, we are indicating to Spark engine to broadcast positive instances
  var crossTot = negativeTot.crossJoin(broadcast(positiveTot))
                             .select(col("pid"), col("nid"),maxRelevanceUDF(col("features"),col("neg_features"),lit(ptot),lit(ntot),lit(n)))
                             .withColumnRenamed(s"UDF(features, neg_features, $ptot, $ntot, $n)","rel")

  var cross_work = crossTot
  if (useCache){
    // positiveTot.cache()
    // negativeTot.cache()
     crossTot.cache()
     cross_work.cache()
   }



   var positive=positiveTot
   var negative=negativeTot

	//Working vaiables for Sij computing
	var p = ptot
	var q = ntot

	if (verbose)
	   println("Creating function with p=" + p + " and q = " + q)

	var top=0

	while (ptot>0){ //2. While there are positive instances

    //Start new term
    var term = Array.fill[Double] (n) (0.0)

    do { //Repeat until there are element in Sij set

      //Compute relevance of R(lk)
      top = cross_work.select("rel").reduce( (ril1,ril2) => maxRelevanceReduce(ril1, ril2,n))(0).asInstanceOf[SparseVector].argmax

      if (verbose) println ("Top: " + top)

  	  if (top > n - 1){
  		  if (verbose) println("x^" + (top-n))
  		  term(top - (n)) = 2.0
  	  }else{
  		  if (verbose) println("x" + top)
  		  term(top) = 1.0
  		  }

        //2.3.3 Select positive and negative instances not satisfied by term's variable (top)
        positive = positive.filter(checkPreserveSijElementUDF(col("features"), lit(top), lit(n)))
        //if (useCache)
        // positive.cache()

        p = positive.count()
        negative = negative.filter(checkPreserveSijElementUDF(col("features"), lit(top), lit(n)))

        //Update cross_join erasing all satisfied positive and anegative instances
        var pos=positive.select(col("pid").as("p_id"))
        var neg=negative.select(col("nid").as("n_id"))

        //Filter the cross-join excluding satisfied positive and negative instances.
        //To make a more performant join a Semi Join is used
        cross_work = cross_work.join(pos,pos.col("p_id")===cross_work.col("pid"),"left_semi").join(neg,neg.col("n_id")===cross_work.col("nid"),"left_semi")
        //cross_work = cross_work.join(broadcast(pos),pos.col("p_id")===cross_work.col("pid"),"left_semi").join(broadcast(neg),neg.col("n_id")===cross_work.col("nid"),"left_semi")

        if (useCache){
          //negative.cache()
          cross_work.cache()
        }

        var newQ = negative.count()

  	  if (q==newQ){
  		throw new Exception("Invalid data found in negative deletion! Training aborted.")
  	  }else{
  		    q = newQ
  	  }
  	  if (verbose) println ("p = "+p+"; q = " + q)

    }while (q > 0 && p>0) //repeat until there are negative instances in set Sij

		//2.4 Add function term
		funz1:+= term
		if (f1.length()>0) f1 += " + "
		for (i <- term.indices){
			if (term(i) == 2.0)
			  f1 += "x"+(i+1)+"^"
			else if (term(i) == 1.0)
			  f1 += "x"+(i+1)
		}

		if (verbose) println("ptot: " + ptot)
    //Update positive instances erasing all the istances satisfying term
    positiveTot = positiveTot.filter(checkPreservePositiveInstancesUDF(col("features"),lit(term)))
    var np = positiveTot.count()

    //Compute the cross Join and calculate relevances of each couple of positive adn negative instances
    crossTot = negativeTot.crossJoin(broadcast(positiveTot))
                               .select(col("pid"), col("nid"),maxRelevanceUDF(col("features"),col("neg_features"),lit(ptot),lit(ntot),lit(n)))
                               .withColumnRenamed(s"UDF(features, neg_features, $ptot, $ntot, $n)","rel")
    cross_work = crossTot
    if (useCache){
       //positiveTot.cache()
       crossTot.cache()
       cross_work.cache()
     }

		if (np>0)
		  positive = positiveTot


		if (verbose){
		  println("ptot: " + ptot)
		  println("np: " + np)
		}

		if (ptot==np){
		  println("break")
		  break
		}

		ptot = np
		if (verbose){
		  println("Positive instances left: " + ptot)
		  println("Function terms computed: " + f1)
		}

		//Reload negative instances
		negative = negativeTot
		if (useCache)
		   //negative.cache()

		q = ntot
		p = ptot
	}
  val trainingTime = (System.nanoTime() - t0)/ 1000000000.0
	if (verbose)
	  println("Generated Function:" + f1 )

	return new BrainScalaModel(geTver,funz1,f1,trainingTime, positiveTotPar.count(), ntot)
  }

}

//Main for testing purpose
object BrainScalaMRCROSSTester{
def  main(args: Array[String]): Unit = {
    if (args.length<2){
      println("Please provide following parameters: <training-set-path> <persistent-model-file-name> <num-partitions-pos> <num-partitions-neg>")
      System.exit(1)
    }

    val spark = SparkSession.builder().appName("Spark-Brain").getOrCreate()
    var instances = spark.read.parquet(args(0))
    var positive = instances.filter("label=1").select("features")
    var negative = instances.filter("label=0").select("features")

    if (args.length>=3 && args(2).toInt>0 )
    {
      positive = positive.repartition(args(2).toInt)
    }
    if (args.length>=4 && args(3).toInt>0)
    {
      negative = negative.repartition(args(3).toInt)
      spark.sqlContext.setConf("spark.default.parallelism",args(3))
      spark.sqlContext.setConf("spark.sql.shuffle.partitions",args(3))
    }

    val t0 = System.nanoTime()

    val f = BrainScalaMRCROSS.fit(positive, negative,false,true)

    val trainingTime = (System.nanoTime() - t0)/ 1000000000.0
    println("Brain Version: "+args(0))
    println("Positive partitions: " + positive.rdd.getNumPartitions)
    println("Negative partitions: " + negative.rdd.getNumPartitions)
    println("f: "+f.funzTxt)
    println("Training Time: "+trainingTime)
    f.saveModel(args(1))
  }
}
