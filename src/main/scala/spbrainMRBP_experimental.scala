//version 1.0  (MRBP 1.0)
//Relevance calculated using MapReduce
//Negative instances are stored in RDDs
//Positive instances are broadcasted as Array[Array[Double]]


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


  object BrainScalaMRBP_experimAssRel{

    def geTver():String = {
      return ("Brain Scala 1.0 exp ass rel- MRBP MapReduce + UDF + Broadcast positive instances (MR2_2) experimental associative Relevance")
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
    def checkErasedSijElement(instance: SparseVector, top: Int, n: Int): Boolean = {
      var indiceTop=0
      var gamma=0.0
      if (top>n-1){
          indiceTop=top-n
          //gamma=false
      }else{
          indiceTop=top
          gamma=1.0
      }
      return (instance(indiceTop)!=gamma)
    }


    //This function return survived and erased instances from Sij (i.e. elements not satisfied by term's variable)
     def getSijInstancesArr(instances: Array[Array[Double]], top: Int, n: Int): Array[Array[Array[Double]]] = {
       var indiceTop=0
       var gamma=0.0
       var erasedInstances = Array.empty[Array[Double]]
       var survivedInstances = Array.empty[Array[Double]]

       if (top>n-1){
           indiceTop=top-n
           //gamma=false
       }else{
           indiceTop=top
           gamma=1.0
       }
       var posIter =instances.iterator
       while (posIter.hasNext)
       {
         var instance = posIter.next()
         if (instance(indiceTop)==gamma)
          survivedInstances:+= instance
         else
          erasedInstances:+= instance
       }
       return Array(survivedInstances, erasedInstances)
    }


  //Relevance compunting. Reduce Task
  //The input consists of two array of the same kind. The output is the sum, elment by element of arrays.
	def maxRelevanceReduce(x: Array[Double], y: Array[Double],n: Int): Array[Double] ={
	  //A most concise and elegant, but a bit slower version is: return (x, y).zipped.map(_ + _)
	  var v = Array.fill[Double] (n*2) (0.0)
	  for (k <- x.indices) {
		    v(k) = x(k) + y(k)
	  }
	  return v
	}

  def subtractVectors(x: Array[Double], y: Array[Double],n: Int): Array[Double] ={
	  //A most concise and elegant, but a bit slower version is: return (x, y).zipped.map(_ + _)
	  var v = Array.fill[Double] (n*2) (0.0)
	  for (k <- x.indices) {
		    v(k) = x(k) - y(k)
	  }
	  return v
	}

	//Relevance computing. Map Task
  //Fow each row ofa dataset computer the relevance of each feature of eache couple negative (the row mapped) with each positive broadcastes and stored in input array.
  //For each couple positive negative, given den = number of total differences betweeb positives and negativeTotD
  //Relevance of i-th feature += 1/den
	def maxRelevanceMap(neg: org.apache.spark.sql.Row, pos:Array[Array[Double]], p: Long, q: Long, n: Int): Array[Double]  = {
		var negative = neg.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]

		var posIter = pos.iterator
		var v = Array.fill[Double] (n*2) (0.0)

		while (posIter.hasNext)
		{
		  var positive = posIter.next()
		  var posIndices = positive.indices
		  var den:Int =0

		  for (k <- posIndices){
			  if (positive(k)!=negative(k)) den +=1
			}

		  var oneOnDen = 1/den.asInstanceOf[Double]
		  //Compute relevance of k-th element of couple i,j
		  for (k <- posIndices){
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
    def maxRelevance(positives: Array[Array[Double]], negatives: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {
      val posList = sc.broadcast(positives)
      val v:Array[Double] =  negatives.rdd.map(row => maxRelevanceMap(row, posList.value,  p, q, n))
          .reduce( (ril1,ril2) => maxRelevanceReduce(ril1,ril2,n))
      v.indices.maxBy(v)
    }

    //return the whole array of relevances
    def maxRelevanceArr(positives: Array[Array[Double]], negatives: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Array[Double] = {
      val posList = sc.broadcast(positives)
      val v:Array[Double] =  negatives.rdd.map(row => maxRelevanceMap(row, posList.value,  p, q, n))
          .reduce( (ril1,ril2) => maxRelevanceReduce(ril1,ril2,n))
      return v
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

	//Register the User Define Function checkPreserveSijElement
	val checkPreserveSijElementUDF = udf(checkPreserveSijElement(_: SparseVector, _: Int, _: Int): Boolean)
  val checkErasedSijElementUDF = udf(checkErasedSijElement(_: SparseVector, _: Int, _: Int): Boolean)

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
  var positiveErased = Array.empty[Array[Double]]
  var negativeErased=negative.limit(1)

	if (verbose)
	   println("Creating function with p=" + p + " and q = " + q)

	//Indice dell'elemento a massima rilevanza per il set Sij
	var top=0
  //Array delle rilevanze
  var topRelevanceArr= Array.empty[Double]
  var posInstancesArr= Array.empty[Array[Array[Double]]]
  var topRelevanceArrPos= Array.empty[Double]
  var topRelevanceArrNeg= Array.empty[Double]
  var topRelevanceArrTmp= Array.empty[Double]
	while (ptot>0){ //2. While there are positive instances

    //Start new term
    var term = Array.fill[Double] (n) (0.0)

    //Build Sij set with remaining elements and compute relevance
    topRelevanceArr = maxRelevanceArr(positive, negative, p, q, n, sc);
    top = topRelevanceArr.indices.maxBy(topRelevanceArr)
    if (top > n - 1){
      if (verbose) println("x^" + (top-n))
      term(top - (n)) = 2.0
    }else{
      if (verbose) println("x" + top)
      term(top) = 1.0
      }
      var i=0

    do { //Repeat until there are element in Sij set
      i=i+1
      if (verbose) println ("Top("+ i +"): " + top)

      //Return positive instances erased
      posInstancesArr = getSijInstancesArr(positive, top, n)
      positiveErased = posInstancesArr(1)

      negativeErased = negative.filter(checkErasedSijElementUDF(col("features"), lit(top), lit(n)))

/*
      topRelevanceArrPos = maxRelevanceArr(positiveErased, negative, p, q, n, sc);
      if (negativeErased.count()>0)
        topRelevanceArrNeg = maxRelevanceArr(positive, negativeErased, p, q, n, sc);
*/
      negative =       negative.filter(checkPreserveSijElementUDF(col("features"), lit(top), lit(n)))
      positive = posInstancesArr(0)
      p = positive.size

      topRelevanceArrTmp=topRelevanceArr
      if (positiveErased.size>0){
        topRelevanceArrPos = maxRelevanceArr(positiveErased, negative, p, q, n, sc);
        topRelevanceArrTmp = subtractVectors(topRelevanceArrTmp,topRelevanceArrPos,n)
      }
      topRelevanceArrNeg = maxRelevanceArr(positive, negativeErased, p, q, n, sc);

      topRelevanceArrTmp = subtractVectors(topRelevanceArrTmp,topRelevanceArrNeg,n)

      top = topRelevanceArrTmp.indices.maxBy(topRelevanceArrTmp)
      topRelevanceArr = topRelevanceArrTmp

      //Calcola porzione del vettore rilevanza relativo ai positivi eliminati
      if (top > n - 1){
  		  if (verbose) println("x^" + (top-n))
  		  term(top - (n)) = 2.0
  	  }else{
  		  if (verbose) println("x" + top)
  		  term(top) = 1.0
  		  }


      println("Negative: "+negative.count())
      println("Negative erased: "+negativeErased.count())
      println("Positive: "+p)
      println("PositiveErased: "+positiveErased.size)

  	  //if (useCache)
  		//  negative.cache()


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
object BrainScalaMRBPAssRelTester{
def  main(args: Array[String]): Unit = {
    if (args.length<2){
      println("Please provide following parameters: brain <training-set-path> <persistent-model-file-name> <num-partitions-pos> <num-partitions-neg>")
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
      negative = negative.repartition(args(3).toInt)

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
