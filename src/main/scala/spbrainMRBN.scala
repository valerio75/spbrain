/**
SP-BRAIN 1.0
@author Valerio Morfino
@version 1.0 - Broadcast Negative

Relevance calculated using MapReduce
Positive instances are stored as DataFrame
Negative instances are broadcasted as Array[Array[Double]]
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
import org.apache.spark.sql.SparkSession
import scala.util.control.Breaks._


  object BrainScalaMRBN{

    def geTver():String = {
      return ("Brain Scala 1.0 - MRBN MapReduce + UDF + Broadcast negative instances (MR3_1)")
    }

    //This function erases from Sij element satisfied by term (given by index top)
    //Given an array of instances in the format of Array[Array[Double]] the function returns not satisfying instances
     def EraseSijElementArr(instances: Array[Array[Double]], top: Int, n: Int): Array[Array[Double]] = {
       var indiceTop=0
       var gamma=0.0
       var survivors = Array.empty[Array[Double]]

       if (top>n-1){
           indiceTop=top-n
           //gamma=0.0
       }else{
           indiceTop=top
           gamma=1.0
       }
       var posIter =instances.iterator
       while (posIter.hasNext)
       {
         var instance = posIter.next()
         if (instance(indiceTop)==gamma)
          survivors:+= instance
       }
       return survivors
    }


  //Relevance compunting. Reduce Task
  //The input consists of two array of the same kind. The output is the sum, elment by element of arrays.
	def maxRelevanceReduceBN(x: Array[Double], y: Array[Double],n: Int): Array[Double] ={
	  //A most concise and elegant, but a bit slower version is: return (x, y).zipped.map(_ + _)
	  var v = Array.fill[Double] (n*2) (0.0)
	  for (k <- x.indices) {
		    v(k) = x(k) + y(k)
	  }
	  return v
	}

	//Relevance computing. Map Task
  //Fow each row ofa dataset computer the relevance of each feature of eache couple negative (the row mapped) with each positive broadcastes and stored in input array.
  //For each couple positive negative, given den = number of total differences betweeb positives and negativeTotD
  //Relevance of i-th feature += 1/den
	def maxRelevanceMapBN(pos: org.apache.spark.sql.Row, neg:Array[Array[Double]], p: Long, q: Long, n: Int): Array[Double]  = {
    var positive = pos.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]  //Tipicamente sarà l'instanza negativa che dovrebbe avere più¢ardinalità

    var negIter =neg.iterator
    var v = Array.fill[Double] (n*2) (0.0)

    while (negIter.hasNext)
    {
      var negative = negIter.next()
      var negIndices = negative.indices
      var den:Int =0
      for (k <- negIndices){
          if (positive(k)!=negative(k)) den +=1
        }
      var oneOnDen = 1/den.asInstanceOf[Double]
		  //Compute relevance of k-th element of couple i,j
      for (k <- negIndices){
        if (positive(k)==1.0 && negative(k)==0.0)
          v(k) +=  oneOnDen
        else if (positive(k)==0.0 && negative(k)==1.0)
          v(k+n) += oneOnDen
      }
    }
    return v
	  }

    //Relevance computing.
    //This task invoke a MAp function to compute the relevance of each comple positive/negative insatnce.
    //The function is composed of four tasks:
    //1) Broadcast positive instances to all workers of the cluster
    //2) A map task that compute the relevance of the features of each negative with each broadcasted positive instance.
    //3) Reduce Task compute a sing vector af overall Relevances of each couple Positive negative
    //4) The index of element with maximum relevance is computed and returned
    def maxRelevanceBN(positives: org.apache.spark.sql.DataFrame, negatives: Array[Array[Double]], p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {
      var negList = sc.broadcast(negatives)
      val v:Array[Double] =  positives.rdd.map(row => maxRelevanceMapBN(row, negList.value,  p, q, n))
          .reduce( (ril1,ril2) => maxRelevanceReduceBN(ril1,ril2,n))
      v.indices.maxBy(v)
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
	var sc = positiveTotPar.rdd.context

	val t0 = System.nanoTime()

	if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
	if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

	//Register the User Define Functions checkPreserveSijElement and checkRetainPositiveInstances in order to use them in Spark SQL context
	val checkPreserveSijElementUDF = udf(BrainUtil.checkPreserveSijElement(_: SparseVector, _: Int, _: Int): Boolean)
  val checkPreservePositiveInstancesUDF = udf(BrainUtil.checkPreservePositiveInstances(_: SparseVector, _: Seq[Double] ): Boolean)

  var f1 = ""
	var funz1 = Array.empty[Array[Double]]

	//Count number of features
	var n= positiveTotPar.head()(0).asInstanceOf[SparseVector].size
	if (verbose)
    println ("n: " + n)

	//Count positive and negative instances
	var ptot=positiveTotPar.count()
	var ntot=negativeTotPar.count()

	//Working vaiables for Sij computing
	var p = ptot
	var q = ntot

	var positiveTot = positiveTotPar
//	if (useCache)
//	   positiveTot.cache()

	//The negative instances will be broadcasted. So the collection is converted in a lighter and simpler format
  var negativeTot = BrainUtil.fromDFOfSparseVectorToArrayOfArrayDouble(negativeTotPar)

	var positive=positiveTot
	var negative=negativeTot

	if (verbose)
	   println("Creating function with p=" + p + " and q = " + q)

	//Indice dell'elemento a massima rilevanza per il set Sij
	var top=0

	while (ptot>0){ //2. While there are positive instances

    //Start new term
    var term = Array.fill[Double] (n) (0.0)

    do { //Repeat until there are element in Sij set

      //Build Sij set with remaining elements and compute relevance
  	  top = maxRelevanceBN(positive, negative, p, q, n, sc); //2.3.1

      if (verbose) println ("Top: " + top)

  	  //
  	  if (top > n - 1){
  		  if (verbose) println("x^" + (top-n))
  		  term(top - (n)) = 2.0
  	  }else{
  		  if (verbose) println("x" + top)
  		  term(top) = 1.0
  		  }

  	  //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
      //Positive are filtered from a local array
  	  negative = EraseSijElementArr(negative, top, n)

      //Negative are filtered using an UDF function of a distributed Dataframe
      positive = positive.filter(checkPreserveSijElementUDF(col("features"), lit(top), lit(n)))

      if (useCache)
  		  positive.cache()

      p=positive.count()

  	  var newQ = negative.size
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
		if (useCache)
      positiveTot.cache()

		var np = positiveTot.count()
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
object BrainScalaMRBNTester{
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

    val f = BrainScalaMRBN.fit(positive, negative,false,true)

    val trainingTime = (System.nanoTime() - t0)/ 1000000000.0
    println("Brain Version: "+args(0))
    println("Positive partitions: " + positive.rdd.getNumPartitions)
    println("Negative partitions: " + negative.rdd.getNumPartitions)
    println("f: "+f.funzTxt)
    println("Training Time: "+trainingTime)
    f.saveModel(args(1))
  }
}
