/**
SP-BRAIN 1.0
@author Valerio Morfino
@version 1.0 - Broadcast Positive

Relevance calculated using MapReduce
Negative instances are stored as DataFrame
Positive instances are broadcasted as Array[Array[Double]]
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


  object BrainScalaMRBP{

    def geTver():String = {
      return ("Brain Scala 1.0 - MRBP MapReduce + UDF + Broadcast positive instances (MR2_2)")
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
	def maxRelevanceReduceBP(x: Array[Double], y: Array[Double],n: Int): Array[Double] ={
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
	def maxRelevanceMapBP(neg: org.apache.spark.sql.Row, pos:Array[Array[Double]], p: Long, q: Long, n: Int): Array[Double]  = {
		var negative = neg.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]

		var posIter = pos.iterator
    var v = new Array[Double](2*n)

		val uncertValue = scala.math.pow(0.5, p + q)
		val uncertValueOne = scala.math.pow(0.5, p + q +1)

		while (posIter.hasNext)
		{
      val posV = new Array[Double](n)
  		var negV = new Array[Double](n)
		  var positive = posIter.next()
		  var posIndices = positive.indices
		  var den:Double = 0

		  for (k <- posIndices){
            if (positive(k) == 1 && negative(k) == 0) {
                posV(k) = 1.0
                //negV(k) = 0
                den += 1.0
            }else if (positive(k) == 0 && negative(k) == 1) {
                //posV(k) = 0
                negV(k) = 1.0
                den += 1.0
            }else if ( (positive(k) == 1 && negative(k) == 0.5) || (positive(k) == 0.5 && negative(k) == 0) ){
                posV(k) = uncertValue
                //negV(k) = 0
                den += uncertValue //vvv20190505
			      }else if (positive(k) == 0.5 && negative(k) == 0.5) {
                posV(k) = uncertValueOne
                negV(k) = uncertValueOne
                den += 2*uncertValueOne //vvv20190505
        		}else if ( (positive(k) == 0 && negative(k) == 0.5) || (positive(k) == 0.5 && negative(k) == 1) ){
        				//posV(k) = 0
        				negV(k) = uncertValue
        				den += uncertValue //vvv20190505
        			}/*else{
                pos(k) = 0
                neg(k) = 0
			        }*/
		  }
		  if (den==0) den=1  //in order to prevent divide by zero error
		  val oneOnDen = 1/den

		  //Compute relevance of k-th element of couple i,j
		  for (k <- posIndices){
			  v(k) +=  (posV(k) * oneOnDen)
			  v(k+n) += (negV(k) * oneOnDen)
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
    def maxRelevanceBP(positives: Array[Array[Double]], negatives: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {
      val posList = sc.broadcast(positives)
      val v:Array[Double] =  negatives.rdd.map(row => maxRelevanceMapBP(row, posList.value,  p, q, n))
          .reduce( (ril1,ril2) => maxRelevanceReduceBP(ril1,ril2,n))
      v.indices.maxBy(v)
    }


  //Filter positive Instaces satisfying the term of function
  //The function works on Array[Array[Double]]
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

  //BRAIN Training
  //The method accept:
  //positiveTotPar as Dataframe containing SparseVector. This is the dataset containing positive Instances  (Cases with class=1.0 or True)
  //negativeTotPar as Dataframe containing SparseVector. This is the dataset containing negative Instances  (Cases with class=0.0 or False)
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

	//Register the User Define Function checkPreserveSijElement
	val checkPreserveSijElementUDF = udf(BrainUtil.checkPreserveSijElement(_: SparseVector, _: Int, _: Int): Boolean)

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

	var negativeTot = negativeTotPar
//	if (useCache)
//	   negativeTot.cache()

	//The posisitve instances will be broadcasted. So the collection is converted in a more light and simple format
	var positiveTot = BrainUtil.fromDFOfSparseVectorToArrayOfArrayDouble(positiveTotPar)

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
  	  top = maxRelevanceBP(positive, negative, p, q, n, sc); //2.3.1

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
  	  positive = EraseSijElementArr(positive, top, n)
  	  p = positive.size
      //Negative are filtered using an UDF function of a distributed Dataframe
  	  negative = negative.filter(checkPreserveSijElementUDF(col("features"), lit(top), lit(n)))

  	  if (useCache)
  		  negative.cache()

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
		positiveTot = filterPositiveInstances(positiveTot,term)

		var np = positiveTot.size
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
		   negative.cache()

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
object BrainScalaMRBPTester{
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

    val f = BrainScalaMRBP.fit(positive, negative,false,true)

    val trainingTime = (System.nanoTime() - t0)/ 1000000000.0
    println("Brain Version: "+args(0))
    println("Positive partitions: " + positive.rdd.getNumPartitions)
    println("Negative partitions: " + negative.rdd.getNumPartitions)
    println("f: "+f.funzTxt)
    println("Training Time: "+trainingTime)
    f.saveModel(args(1))
  }
}
