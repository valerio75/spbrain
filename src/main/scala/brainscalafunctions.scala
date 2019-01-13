/**
SP-BRAIN 1.0
@author Valerio Morfino
@version 1.0 

Utility functions
*/

package brain.scala
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.broadcast
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.SparkSession

import scala.util.control.Breaks._

  object BrainScala{

    def geTver():String = {
      return ("Brain Scala 0.2 MapReduce + UDF. Modified RElevance Map Function: eliminated div p div q in formula, because they are not needed in RElevance computing")
    }

    //Used for positive and negative Deletion
    //The function return true if instance non have to be deleted
    //Usato per l'eliminazione delle coppie Sij
    def checkPreserveInstance(instance: SparseVector, top: Int, n: Int): Boolean = {
      var indiceTop=0
      var gamma=0.0
      if (top>n-1){
          indiceTop=top-n
          //gamma=false #Inutile perché la variabile è già inizializzata a false
      }else{
          indiceTop=top
          gamma=1.0
      }
      return (instance(indiceTop)==gamma)
    }

    //UDF per deletion istanze positive che soddisfano la funzione
    //Quì c'è qualcosa da ottimizzare rispetto al termine rappresenatato come stringa che viene poi splittato in array
    def checkRetainPositiveInstances(instance: SparseVector, t: String): Boolean ={
      var term:Array[String] = t.split("")
      //var instance = pos(0).asInstanceOf [SparseVector]

      for (i <- instance.toArray.indices){
        if ((instance(i)==1.0 && term(i)=="0") || (instance(i)==0.0 && term(i)=="1")) {
          //println (instance(i) + "--" + term(i) + "---" + (instance(i)==1.0) + "----" + (term(i)=='0'))
          //println("Ritorno True")
          return true
          }
      }
      //println("Ritorno False")
      return false
    }
    //checkRetainPositive = udf(checkRetainPositive_, BooleanType())

    //#Register UDF to be used in Dataframes for positive and negaive deletion
    //checkPreserveInstance = udf(checkPreserve_, BooleanType())

    def maxRilevanzaMap(row: org.apache.spark.sql.Row, p: Long, q: Long, n: Int): org.apache.spark.ml.linalg.SparseVector  = {
        var positive = row.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]
        var negative = row.getAs("neg_features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]

        var den=0.0

        //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
        //Conta numero di differenze tra positivo e negativo
        for (k <- positive.toArray.indices){
            if (positive(k)!=negative(k)) den +=1
          }
        var v = Array.fill[Double] (n*2) (0.0)

        //Calcola la rilevanza del k-esimo elemento della coppia i,j
        for (k <- positive.toArray.indices){
          //println("k = " + k + " ((1 / den) / q) / p=" + (((1 / den) / q) / p) )
          if (positive(k)==1.0 && negative(k)==0.0)
            //v(k) +=  ((1 / den) / q) / p  //Attenzione ho reso implicito il cast a float. Verificare se è ok
            v(k) +=  (1 / den)
          else if (positive(k)==0.0 && negative(k)==1.0)
            //v(k+n) += ((1 / den) / q) / p //Attenzione ho reso implicito il cast a float. Verificare se è ok
            v(k+n) += (1 / den)
        }

        return (new org.apache.spark.ml.linalg.DenseVector(v)).toSparse
      }

      def maxRilevanzaMapArr(row: org.apache.spark.sql.Row, p: Long, q: Long, n: Int): Array[Double]  = {
          var positive = row.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]
          var negative = row.getAs("neg_features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]

          var den=0.0

          //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
          //Conta numero di differenze tra positivo e negativo
          for (k <- positive.toArray.indices){
              if (positive(k)!=negative(k)) den +=1
            }
          var v = Array.fill[Double] (n*2) (0.0)

          //Calcola la rilevanza del k-esimo elemento della coppia i,j
          for (k <- positive.toArray.indices){
            //println("k = " + k + " ((1 / den) / q) / p=" + (((1 / den) / q) / p) )
            if (positive(k)==1.0 && negative(k)==0.0)
              //v(k) +=  ((1 / den) / q) / p  //Attenzione ho reso implicito il cast a float. Verificare se è ok
              v(k) +=  (1 / den)
            else if (positive(k)==0.0 && negative(k)==1.0)
              //v(k+n) += ((1 / den) / q) / p //Attenzione ho reso implicito il cast a float. Verificare se è ok
              v(k+n) += (1 / den)
          }

          return v
        }

        def maxRilevanzaUDF(positive:SparseVector, negative:SparseVector, p: Long, q: Long, n: Int): SparseVector  = {
            //var positive = row.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]
            //var negative = row.getAs("neg_features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]

            var den=0.0
            //println("****************VVVVVVVVVVVVV - maxRilevanzaMapArr - VVVVVVVVVVVVVV********************************************")
            //print("*")
            //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
            //Conta numero di differenze tra positivo e negativo
            for (k <- positive.toArray.indices){
                if (positive(k)!=negative(k)) den +=1
              }
            var v = Array.fill[Double] (n*2) (0.0)

            //Calcola la rilevanza del k-esimo elemento della coppia i,j
            for (k <- positive.toArray.indices){
              //println("k = " + k + " ((1 / den) / q) / p=" + (((1 / den) / q) / p) )
              if (positive(k)==1.0 && negative(k)==0.0)
                v(k) +=  ((1 / den) / q) / p  //Attenzione ho reso implicito il cast a float. Verificare se è ok
              else if (positive(k)==0.0 && negative(k)==1.0)
                v(k+n) += ((1 / den) / q) / p //Attenzione ho reso implicito il cast a float. Verificare se è ok
            }

            return (new org.apache.spark.ml.linalg.DenseVector(v)).toSparse
          }

        def maxRilevanzaMapRow(row: org.apache.spark.sql.Row, p: Long, q: Long, n: Int): org.apache.spark.sql.Row  = {
            var positive = row.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]
            var negative = row.getAs("neg_features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]

            var den=0.0

            //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
            //Conta numero di differenze tra positivo e negativo
            for (k <- positive.toArray.indices){
                if (positive(k)!=negative(k)) den +=1
              }
            var v = Array.fill[Double] (n*2) (0.0)

            //Calcola la rilevanza del k-esimo elemento della coppia i,j
            for (k <- positive.toArray.indices){
              //println("k = " + k + " ((1 / den) / q) / p=" + (((1 / den) / q) / p) )
              if (positive(k)==1.0 && negative(k)==0.0)
                v(k) +=  ((1 / den) / q) / p  //Attenzione ho reso implicito il cast a float. Verificare se è ok
              else if (positive(k)==0.0 && negative(k)==1.0)
                v(k+n) += ((1 / den) / q) / p //Attenzione ho reso implicito il cast a float. Verificare se è ok
            }

            return Row(row(0), row(1), row(2), row(3), v)
          }

      def maxRilevanzaRed(x: org.apache.spark.ml.linalg.SparseVector, y: org.apache.spark.ml.linalg.SparseVector,n: Int): org.apache.spark.ml.linalg.SparseVector ={
        //Alternativa in una riga che però mi sembra più lenta:
        //(x, y).zipped.map(_ + _)
        var v = Array.fill[Double] (n*2) (0.0)
        for (k <- x.toArray.indices) {
          v(k) = x(k) + y(k)
        }
        return (new org.apache.spark.ml.linalg.DenseVector(v)).toSparse
      }

      def maxRilevanzaRedRow(xx: org.apache.spark.sql.Row, yy: org.apache.spark.sql.Row,n: Int): org.apache.spark.sql.Row ={
        //Alternativa in una riga che però mi sembra più lenta:
        //(x, y).zipped.map(_ + _)
        var x = xx.get(0).asInstanceOf[org.apache.spark.ml.linalg.SparseVector]
        var y = yy.get(0).asInstanceOf[org.apache.spark.ml.linalg.SparseVector]

        var v = Array.fill[Double] (n*2) (0.0)
        for (k <- x.toArray.indices) {
          v(k) = x(k) + y(k)
        }
        return Row ((new org.apache.spark.ml.linalg.DenseVector(v)).toSparse)
      }
      def maxRilevanzaMR(positives: org.apache.spark.sql.DataFrame, neg: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int): Int = {
        var negatives = neg.withColumnRenamed("features","neg_features")
        var cross = negatives.crossJoin(positives)

        var rdd =cross.rdd.map(row => maxRilevanzaMap(row, p, q, n))
        var v:org.apache.spark.ml.linalg.SparseVector = rdd.reduce( (ril1,ril2) => maxRilevanzaRed(ril1,ril2,n))
        return v.argmax //Ritorna posizione del massimale. Verificare che vada bene come massimo dell'algoritmo **********
        //v.toArray.indexOf(v.toArray.max) //Ottimizzare
      }

      def brainTrainingMR_ds(positiveTotParDs: Dataset[SparseVector] , negativeTotDs: Dataset[SparseVector], verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true): String = {

        brainTrainingMR(positiveTotParDs.toDF(), negativeTotDs.toDF())
      }

      def brainTrainingMR(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true): String = {
        val t0 = System.nanoTime()

      //  var positiveTotPar = positiveTotParDs.toDF()
        var negativeTot = negativeTotPar

        var positiveTot = positiveTotPar

        if (positiveTot.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
        if (negativeTot.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

        //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
        val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)
        val checkRetainPositiveUDF = udf(checkRetainPositiveInstances(_: SparseVector, _: String ): Boolean)

        var f1 = ""
        var funz1 = Array.empty[String]

        if (useCache){
          positiveTot = positiveTot.cache()
          negativeTot = negativeTot.cache()
        }
        //***Verificare che effettivamente avvenga la copia
        var positive=positiveTot
        var negative=negativeTot

        //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
        var n= positiveTot.head()(0).asInstanceOf[SparseVector].size
        println ("n: " + n)

        //numero totale dei positivi e negativi
        var ptot=positiveTot.count()
        var ntot=negativeTot.count()
        //numero dei positivi e negativi utilizzati per calcolare gli Si
        var p = ptot
        var q = ntot

        if (verbose)
           println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

        //Initizialize literal Function
        //var term = "*" * n
        var term = Array.fill[String] (n) ("*")

        //Indice dell'elemento a massima rilevanza per il set Sij
        var top=0

        while (ptot>0){ //sez. 2 Finché ci sono positivi
          //Calcola rilevanza massima
          top = maxRilevanzaMR(positive, negative, p, q, n); //2.3.1
          if (verbose) println ("Top: " + top)

          //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
          if (top > n - 1){
              if (verbose) println("x^" + (top-n))
              term(top - (n)) = "0"
          }else{
              if (verbose) println("x" + top)
              term(top) = "1"
              }

          //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
          positive = positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
          if (useCache)
            positive = positive.cache()
          p = positive.count()
          negative = negative.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
          if (useCache)
            negative = negative.cache()
          var newQ = negative.count()
          if (q==newQ){
            throw new Exception("Invalid data found in negative deletion! Training aborted.")
          }else{
            q = newQ
          }
          if (verbose) println ("p = "+p+"; q = " + q)

          /*
          Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
          Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
          come fosse un repeat until utilizzando i negativi superstiti come indicatore
          */
          if (q <= 0){
            //2.4 Add function term
            funz1:+ term.mkString("")
            if (f1.length()>0) f1 += " + "
            for (i <- term.indices){
                if (term(i) == "0")
                  f1 += "x"+(i+1)+"^"
                else if (term(i) == "1")
                  f1 += "x"+(i+1)
            }


            //Elimina i termini positivi che soddisfano la funzione,
            //o meglio seleziona solo i termini non soddisfatti
            //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
            println("ptot: " + ptot)
            var t = term.mkString("")
            positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t)))
            var np = positiveTot.count()

            if (np>0)
              positive = positiveTot

            if (useCache)
              positive = positive.cache()

            if (verbose){
              println("ptot: " + ptot)
              println("np: " + np)
            }

            //****Verificare l'effettiva utilità di questa istruzione*****
            if (ptot==np){
              println("break del while")
              break
            }

            ptot = np
            if (verbose){
              println("positivi rimasti: " + ptot)
              println("Funzione parziale ottenuta: " + f1)
            }

            //Rilegge le istanze negative
            negative = negativeTot
            if (useCache)
              negative = negative.cache()

            q = ntot
            p = ptot
            term = Array.fill[String] (n) ("*")
          } //end della if q<=0
        }

        if (verbose)
          println("Generated Function:" + f1 )

        if (printElapsed)
          println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

        return f1


      }

      def brainTrainingV(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true): String = {
        val t0 = System.nanoTime()
        //Prepare positive and negative Instances
        //Prepara le istanze positive e negative
        if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
        if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

        //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
        val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)
        val checkRetainPositiveUDF = udf(checkRetainPositiveInstances(_: SparseVector, _: String ): Boolean)
        val maxRilevanza = udf(brain.scala.BrainScala.maxRilevanzaUDF(_: SparseVector, _: SparseVector, _: Long, _: Long, _: Int): SparseVector)

        var f1 = ""
        var funz1 = Array.empty[String]

        //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
        var n = positiveTotPar.head()(0).asInstanceOf[SparseVector].size
        println ("n: " + n)

        //numero totale dei positivi e negativi
        var ptot=positiveTotPar.count()
        var ntot=negativeTotPar.count()

        //Aggiunge alle istanze positive ed alle negative uan colonna con ID che servirà per il successivo left semi-join per escludere le isatnze eliminate
        var positiveTot = positiveTotPar.withColumn("pid",monotonically_increasing_id())
        var negativeTot = negativeTotPar.withColumnRenamed("features","neg_features").withColumn("nid",monotonically_increasing_id())

        //Crea il Cross-join ossia un prodotto cartesiano tra le istanze positive e negative. In totale avremo ptot*ntot righw
        //<TODO>***NOTA INTERESSANTE SULL'OTTIMIZZAZIONE: Contanto se sono più le istanze positive o quelle negative, si può decidere quale dei due DataFrame vada in broadcast
        var cross = negativeTot.crossJoin(positiveTot)
        //Il calcolo delle rilevanze ij avviene una sola volta per tutta l'elaborazione
        var crossTot = cross.select(col("pid"), col("nid"),maxRilevanza(col("features"),col("neg_features"),lit(ptot),lit(ntot),lit(n)))
                            .withColumnRenamed(s"UDF(features, neg_features, $ptot, $ntot, $n)","rel")
        //crossTot.write.parquet("temp")
        //crossTot = sess.read.parquet("temp")

        crossTot.cache()
        if (useCache){
          positiveTot = positiveTot.cache()
          negativeTot = negativeTot.cache()
          crossTot = crossTot.cache()
        }

        var positive=positiveTot
        var negative=negativeTot
        //var cross_work = crossTot.select(col("features"), col("neg_features"),col("pid"), col("nid"),col("rel"))
        var cross_work = crossTot.select(col("pid"), col("nid"),col("rel"))

        var p = ptot
        var q = ntot

        if (verbose)
           println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

        //Initizialize literal Function
        //var term = "*" * n
        var term = Array.fill[String] (n) ("*")

        //Indice dell'elemento a massima rilevanza per il set Sij
        var top=0

        while (ptot>0){ //sez. 2 Finché ci sono positivi
          //Calcola rilevanza massima
          //top = maxRilevanzaMR(positive, negative, p, q, n); //2.3.1
          top = cross_work.select("rel").rdd.
                                            map(line => line(0).asInstanceOf[SparseVector]).
                                            reduce( (ril1,ril2) => brain.scala.BrainScala.maxRilevanzaRed(ril1, ril2,n)).argmax

          if (verbose) println ("Top: " + top)

          //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
          if (top > n - 1){
              if (verbose) println("x^" + (top-n))
              term(top - (n)) = "0"
          }else{
              if (verbose) println("x" + top)
              term(top) = "1"
              }

          //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
          positive = positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
          if (useCache)
            positive = positive.cache()
          p = positive.count()
          negative = negative.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))


          //Update cross_join
          var pos=positive.select(col("pid").as("p_id"))
          var neg=negative.select(col("nid").as("n_id"))
          cross_work = cross_work.join(pos,pos.col("p_id")===cross_work.col("pid"),"left_semi").join(neg,neg.col("n_id")===cross_work.col("nid"),"left_semi")

          if (useCache)
            negative = negative.cache()

          var newQ = negative.count()
          if (q==newQ){
            throw new Exception("Invalid data found in negative deletion! Training aborted.")
          }else{
            q = newQ
          }
          if (verbose) println ("p = "+p+"; q = " + q)

          /*
          Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
          Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
          come fosse un repeat until utilizzando i negativi superstiti come indicatore
          */
          if (q <= 0){
            //2.4 Add function term
            funz1:+ term.mkString("")
            if (f1.length()>0) f1 += " + "
            for (i <- term.indices){
                if (term(i) == "0")
                  f1 += "x"+(i+1)+"^"
                else if (term(i) == "1")
                  f1 += "x"+(i+1)
            }


            //Elimina i termini positivi che soddisfano la funzione,
            //o meglio seleziona solo i termini non soddisfatti
            //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
            println("ptot: " + ptot)
            var t = term.mkString("")
            positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t)))
            var np = positiveTot.count()

            //Aggiorna il cross_join eliminando le istanze negative eliminate
            var posTot = positiveTot.select(col("pid").as("p_id"))
            crossTot = crossTot.join(posTot,posTot.col("p_id")===crossTot.col("pid"),"left_semi")
            //crossTot.cache()
            cross_work = crossTot.select(col("pid"), col("nid"),col("rel"))

            if (np>0)
              positive = positiveTot

            if (useCache)
              positive = positive.cache()

            if (verbose){
              println("ptot: " + ptot)
              println("np: " + np)
            }



            //****Verificare l'effettiva utilità di questa istruzione*****
            if (ptot==np){
              println("break del while")
              break
            }

            ptot = np
            if (verbose){
              println("positivi rimasti: " + ptot)
              println("Funzione parziale ottenuta: " + f1)
            }

            //Rilegge le istanze negative
            negative = negativeTot
            if (useCache)
              negative = negative.cache()

            q = ntot
            p = ptot
            term = Array.fill[String] (n) ("*")
          } //end della if q<=0
        }

        if (verbose)
          println("Generated Function:" + f1 )

        if (printElapsed)
          println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

        return f1


      }

    def brainTrainingV0_1(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true): String = {
        val t0 = System.nanoTime()
        //Prepare positive and negative Instances
        //Prepara le istanze positive e negative
        if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
        if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

        //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
        val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)
        val checkRetainPositiveUDF = udf(checkRetainPositiveInstances(_: SparseVector, _: String ): Boolean)
        val maxRilevanza = udf(brain.scala.BrainScala.maxRilevanzaUDF(_: SparseVector, _: SparseVector, _: Long, _: Long, _: Int): SparseVector)

        var f1 = ""
        var funz1 = Array.empty[String]

        //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
        var n = positiveTotPar.head()(0).asInstanceOf[SparseVector].size
        println ("n: " + n)

        //numero totale dei positivi e negativi
        var ptot=positiveTotPar.count()
        var ntot=negativeTotPar.count()

        //Aggiunge alle istanze positive ed alle negative uan colonna con ID che servirà per il successivo left semi-join per escludere le isatnze eliminate
        var positiveTot = positiveTotPar.withColumn("pid",monotonically_increasing_id())
        var negativeTot = negativeTotPar.withColumnRenamed("features","neg_features").withColumn("nid",monotonically_increasing_id())

        //Crea il Cross-join ossia un prodotto cartesiano tra le istanze positive e negative. In totale avremo ptot*ntot righw
        //<TODO>***NOTA INTERESSANTE SULL'OTTIMIZZAZIONE: Contanto se sono più le istanze positive o quelle negative, si può decidere quale dei due DataFrame vada in broadcast
        //var cross = negativeTot.crossJoin(positiveTot)
        //Il calcolo delle rilevanze ij avviene una sola volta per tutta l'elaborazione
        var crossTot = negativeTot.crossJoin(positiveTot).select(col("pid"), col("nid"),maxRilevanza(col("features"),col("neg_features"),lit(ptot),lit(ntot),lit(n)))
                            .withColumnRenamed(s"UDF(features, neg_features, $ptot, $ntot, $n)","rel")


        //crossTot.cache()
        if (useCache){
          positiveTot.cache()
          negativeTot.cache()
          crossTot.cache()
        }

        var positive=positiveTot
        var negative=negativeTot

        //var cross_work = crossTot.select(col("pid"), col("nid"),col("rel"))
	       var cross_work = crossTot
         if (useCache){
           cross_work.cache()
         }
        var p = ptot
        var q = ntot

        if (verbose)
           println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

        //Initizialize literal Function
        //var term = "*" * n
        var term = Array.fill[String] (n) ("*")

        //Indice dell'elemento a massima rilevanza per il set Sij
        var top=0

        while (ptot>0){ //sez. 2 Finché ci sono positivi
          //Calcola rilevanza massima
          //top = maxRilevanzaMR(positive, negative, p, q, n); //2.3.1
          top = cross_work.select("rel").rdd.
                                            map(line => line(0).asInstanceOf[SparseVector]).
                                            reduce( (ril1,ril2) => brain.scala.BrainScala.maxRilevanzaRed(ril1, ril2,n)).argmax

          if (verbose) println ("Top: " + top)

          //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
          if (top > n - 1){
              if (verbose) println("x^" + (top-n))
              term(top - (n)) = "0"
          }else{
              if (verbose) println("x" + top)
              term(top) = "1"
              }

          //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
          positive = positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
          if (useCache)
            positive = positive.cache()
          p = positive.count()
          negative = negative.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))


          //Update cross_join
          var pos=positive.select(col("pid").as("p_id"))
          var neg=negative.select(col("nid").as("n_id"))
          cross_work = cross_work.join(pos,pos.col("p_id")===cross_work.col("pid"),"left_semi").join(neg,neg.col("n_id")===cross_work.col("nid"),"left_semi")

          if (useCache){
            cross_work.cache()
            negative.cache()
          }



          var newQ = negative.count()
          if (q==newQ){
            throw new Exception("Invalid data found in negative deletion! Training aborted.")
          }else{
            q = newQ
          }
          if (verbose) println ("p = "+p+"; q = " + q)

          /*
          Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
          Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
          come fosse un repeat until utilizzando i negativi superstiti come indicatore
          */
          if (q <= 0){
            //2.4 Add function term
            funz1:+ term.mkString("")
            if (f1.length()>0) f1 += " + "
            for (i <- term.indices){
                if (term(i) == "0")
                  f1 += "x"+(i+1)+"^"
                else if (term(i) == "1")
                  f1 += "x"+(i+1)
            }


            //Elimina i termini positivi che soddisfano la funzione,
            //o meglio seleziona solo i termini non soddisfatti
            //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
            println("ptot: " + ptot)
            var t = term.mkString("")
            positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t)))
            var np = positiveTot.count()

            //Aggiorna il cross_join eliminando le istanze negative eliminate
            var posTot = positiveTot.select(col("pid").as("p_id"))
            crossTot = crossTot.join(posTot,posTot.col("p_id")===crossTot.col("pid"),"left_semi")
            //crossTot.cache()
            cross_work = crossTot.select(col("pid"), col("nid"),col("rel"))

            if (np>0)
              positive = positiveTot

            if (useCache)
              positive = positive.cache()

            if (verbose){
              println("ptot: " + ptot)
              println("np: " + np)
            }



            //****Verificare l'effettiva utilità di questa istruzione*****
            if (ptot==np){
              println("break del while")
              break
            }

            ptot = np
            if (verbose){
              println("positivi rimasti: " + ptot)
              println("Funzione parziale ottenuta: " + f1)
            }

            //Rilegge le istanze negative
            negative = negativeTot
            if (useCache)
              negative = negative.cache()

            q = ntot
            p = ptot
            term = Array.fill[String] (n) ("*")
          } //end della if q<=0
        }

        if (verbose)
          println("Generated Function:" + f1 )

        if (printElapsed)
          println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

        return f1


      }
      def brainTrainingV1(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true, sess: org.apache.spark.sql.SparkSession): String = {
        val t0 = System.nanoTime()
        //Prepare positive and negative Instances
        //Prepara le istanze positive e negative
        if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
        if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

        //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
        val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)
        val checkRetainPositiveUDF = udf(checkRetainPositiveInstances(_: SparseVector, _: String ): Boolean)
        val maxRilevanza = udf(brain.scala.BrainScala.maxRilevanzaUDF(_: SparseVector, _: SparseVector, _: Long, _: Long, _: Int): SparseVector)

        var f1 = ""
        var funz1 = Array.empty[String]

        //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
        var n = positiveTotPar.head()(0).asInstanceOf[SparseVector].size
        println ("n: " + n)

        //numero totale dei positivi e negativi
        var ptot=positiveTotPar.count()
        var ntot=negativeTotPar.count()

        //Aggiunge alle istanze positive ed alle negative uan colonna con ID che servirà per il successivo left semi-join per escludere le isatnze eliminate
        //var positiveTot = positiveTotPar.withColumn("pid",monotonically_increasing_id())
        //var negativeTot = negativeTotPar.withColumnRenamed("features","neg_features").withColumn("nid",monotonically_increasing_id())

        //Crea il Cross-join ossia un prodotto cartesiano tra le istanze positive e negative. In totale avremo ptot*ntot righw
        //<TODO>***NOTA INTERESSANTE SULL'OTTIMIZZAZIONE: Contanto se sono più le istanze positive o quelle negative, si può decidere quale dei due DataFrame vada in broadcast
        var cross = negativeTotPar.withColumnRenamed("features","neg_features").crossJoin(positiveTotPar)
        //Il calcolo delle rilevanze ij avviene una sola volta per tutta l'elaborazione
        var crossTot = cross.select(col("features"), col("neg_features"), maxRilevanza(col("features"),col("neg_features"),lit(ptot),lit(ntot),lit(n)))
                            .withColumnRenamed(s"UDF(features, neg_features, $ptot, $ntot, $n)","rel")
        //crossTot.write.parquet("temp")
        //crossTot = sess.read.parquet("temp")

        crossTot.cache()
        /*if (useCache){
          positiveTot = positiveTot.cache()
          negativeTot = negativeTot.cache()
          crossTot = crossTot.cache()
        }*/

        //var positive=positiveTot
        //var negative=negativeTot
        //var cross_work = crossTot.select(col("features"), col("neg_features"),col("pid"), col("nid"),col("rel"))
        //var cross_work = crossTot.select(col("pid"), col("nid"),col("rel"))
        var cross_work = crossTot

        var p = ptot
        var q = ntot

        if (verbose)
           println("Creo funzione con p=" + p + "; q = " + q)

        //Initizialize literal Function
        //var term = "*" * n
        var term = Array.fill[String] (n) ("*")

        //Indice dell'elemento a massima rilevanza per il set Sij
        var top=0
        //var cr =Long
        var ctot = crossTot.count()
        while (ctot>0){ //sez. 2 Finché ci sono positivi
          //println(crossTot.count())
          //Calcola rilevanza massima
          //top = maxRilevanzaMR(positive, negative, p, q, n); //2.3.1
          top = cross_work.select("rel").rdd.
                                            map(line => line(0).asInstanceOf[SparseVector]).
                                            reduce( (ril1,ril2) => brain.scala.BrainScala.maxRilevanzaRed(ril1, ril2,n)).argmax

          if (verbose) println ("Top: " + top)

          //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
          if (top > n - 1){
              if (verbose) println("x^" + (top-n))
              term(top - (n)) = "0"
          }else{
              if (verbose) println("x" + top)
              term(top) = "1"
              }

          //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
          cross_work = cross_work.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
        //  if (useCache)
        //    positive = positive.cache()
          //p = positive.count()
          cross_work = cross_work.filter(checkPreserveInstanceUDF(col("neg_features"), lit(top), lit(n)))


          //Update cross_join
          //var pos=positive.select(col("pid").as("p_id"))
          //var neg=negative.select(col("nid").as("n_id"))
          //cross_work = cross_work.join(pos,pos.col("p_id")===cross_work.col("pid"),"left_semi").join(neg,neg.col("n_id")===cross_work.col("nid"),"left_semi")

        //  if (useCache)
        //    negative = negative.cache()

          //var newQ = negative.count()
          /*
          if (q==newQ){
            throw new Exception("Invalid data found in negative deletion! Training aborted.")
          }else{
            q = newQ
          }
          if (verbose) println ("p = "+p+"; q = " + q)
          */

          /*
          Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
          Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
          come fosse un repeat until utilizzando i negativi superstiti come indicatore
          */
          //println("cross_work" + cross_work.count())
          if (cross_work.count() <= 0){
            //2.4 Add function term
            funz1:+ term.mkString("")
            if (f1.length()>0) f1 += " + "
            for (i <- term.indices){
                if (term(i) == "0")
                  f1 += "x"+(i+1)+"^"
                else if (term(i) == "1")
                  f1 += "x"+(i+1)
            }


            //Elimina i termini positivi che soddisfano la funzione,
            //o meglio seleziona solo i termini non soddisfatti
            //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******

            var t = term.mkString("")
            //println("pre: " + crossTot.count()+ " ")
            crossTot = crossTot.filter(checkRetainPositiveUDF(col("features"),lit(t)))
            cross_work = crossTot
            ctot = crossTot.count()
            println("post: " + ctot)
            //var np = positiveTot.count()
            /*
            //Aggiorna il cross_join eliminando le istanze negative eliminate
            var posTot = positiveTot.select(col("pid").as("p_id"))
            crossTot = crossTot.join(posTot,posTot.col("p_id")===crossTot.col("pid"),"left_semi")
            //crossTot.cache()
            cross_work = crossTot.select(col("pid"), col("nid"),col("rel"))

            if (np>0)
              positive = positiveTot
            */
          //  if (useCache)
          //    positive = positive.cache()

            if (verbose){
              println("ptot: " + ptot)
            //  println("np: " + np)
            }



            //****Verificare l'effettiva utilità di questa istruzione*****
            /*
            if (ptot==np){
              println("break del while")
              break
            }

            ptot = np
            if (verbose){
              println("positivi rimasti: " + ptot)
              println("Funzione parziale ottenuta: " + f1)
            }
            */
            //Rilegge le istanze negative
            //negative = negativeTot
          //  if (useCache)
          //    negative = negative.cache()

            //q = ntot
            //p = ptot
            term = Array.fill[String] (n) ("*")
          } //end della if q<=0
        }

        if (verbose)
          println("Generated Function:" + f1 )

        if (printElapsed)
          println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

        return f1
      }

/*******************************MR2 Elaborazione su istance negativr con tutte le positive in broadcast************************************************************************/

    def maxRilevanzaMap2(neg: org.apache.spark.sql.Row, pos:java.util.List[org.apache.spark.sql.Row], p: Long, q: Long, n: Int): org.apache.spark.ml.linalg.SparseVector  = {
        var negative = neg.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]  //Tipicamente sarà l'instanza negativa che dovrebbe avere più¢ardinalità


        var posIter =pos.iterator
        var v = Array.fill[Double] (n*2) (0.0)

        while (posIter.hasNext)
        {
          var positive = posIter.next()(0).asInstanceOf[SparseVector].toArray
          var posIndices = positive.indices
          var den=0.0
          //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
          //Conta numero di differenze tra positivo e negativo
          for (k <- posIndices){
              if (positive(k)!=negative(k)) den +=1
            }

          //Calcola la rilevanza del k-esimo elemento della coppia i,j
          for (k <- posIndices){
            //println("k = " + k + " ((1 / den) / q) / p=" + (((1 / den) / q) / p) )
            if (positive(k)==1.0 && negative(k)==0.0)
              //v(k) +=  ((1 / den) / q) / p  //Attenzione ho reso implicito il cast a float. Verificare se è ok
              v(k) +=  (1 / den)
            else if (positive(k)==0.0 && negative(k)==1.0)
              //v(k+n) += ((1 / den) / q) / p //Attenzione ho reso implicito il cast a float. Verificare se è ok
              v(k+n) += (1 / den)
          }
        }
        return (new org.apache.spark.ml.linalg.DenseVector(v)).toSparse
      }

      def maxRilevanzaMR2(positives: org.apache.spark.sql.DataFrame, negatives: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {
        //var negatives = neg.withColumnRenamed("features","neg_features")
        //var cross = negatives.crossJoin(positives)
        var posList = sc.broadcast(positives.takeAsList(p.asInstanceOf[Int]))
        //var posList = positives.takeAsList(p.asInstanceOf[Int])
        var rdd =negatives.rdd.map(row => maxRilevanzaMap2(row, posList.value,  p, q, n))
        //var rdd =negatives.rdd.map(row => maxRilevanzaMap2(row, posList,  p, q, n))
        var v:org.apache.spark.ml.linalg.SparseVector = rdd.reduce( (ril1,ril2) => maxRilevanzaRed(ril1,ril2,n))
        return v.argmax //Ritorna posizione del massimale. Verificare che vada bene come massimo dell'algoritmo **********
        //v.toArray.indexOf(v.toArray.max) //Ottimizzare
      }

      def brainTrainingMR2(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true, sc: org.apache.spark.SparkContext): String = {
        val t0 = System.nanoTime()

      //  var positiveTotPar = positiveTotParDs.toDF()
        var negativeTot = negativeTotPar

        var positiveTot = positiveTotPar

        if (positiveTot.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
        if (negativeTot.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

        //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
        val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)
        val checkRetainPositiveUDF = udf(checkRetainPositiveInstances(_: SparseVector, _: String ): Boolean)

        var f1 = ""
        var funz1 = Array.empty[String]

        if (useCache){
          positiveTot = positiveTot.cache()
          negativeTot = negativeTot.cache()
        }
        //***Verificare che effettivamente avvenga la copia
        var positive=positiveTot
        var negative=negativeTot

        //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
        var n= positiveTot.head()(0).asInstanceOf[SparseVector].size
        println ("n: " + n)

        //numero totale dei positivi e negativi
        var ptot=positiveTot.count()
        var ntot=negativeTot.count()
        //numero dei positivi e negativi utilizzati per calcolare gli Si
        var p = ptot
        var q = ntot

        if (verbose)
           println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

        //Initizialize literal Function
        //var term = "*" * n
        var term = Array.fill[String] (n) ("*")

        //Indice dell'elemento a massima rilevanza per il set Sij
        var top=0

        while (ptot>0){ //sez. 2 Finché ci sono positivi
          //Calcola rilevanza massima
          top = maxRilevanzaMR2(positive, negative, p, q, n, sc); //2.3.1
          if (verbose) println ("Top: " + top)

          //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
          if (top > n - 1){
              if (verbose) println("x^" + (top-n))
              term(top - (n)) = "0"
          }else{
              if (verbose) println("x" + top)
              term(top) = "1"
              }

          //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
          positive = positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
          if (useCache)
            positive = positive.cache()
          p = positive.count()
          negative = negative.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
          if (useCache)
            negative = negative.cache()
          var newQ = negative.count()
          if (q==newQ){
            throw new Exception("Invalid data found in negative deletion! Training aborted.")
          }else{
            q = newQ
          }
          if (verbose) println ("p = "+p+"; q = " + q)

          /*
          Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
          Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
          come fosse un repeat until utilizzando i negativi superstiti come indicatore
          */
          if (q <= 0){
            //2.4 Add function term
            funz1:+ term.mkString("")
            if (f1.length()>0) f1 += " + "
            for (i <- term.indices){
                if (term(i) == "0")
                  f1 += "x"+(i+1)+"^"
                else if (term(i) == "1")
                  f1 += "x"+(i+1)
            }


            //Elimina i termini positivi che soddisfano la funzione,
            //o meglio seleziona solo i termini non soddisfatti
            //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
            println("ptot: " + ptot)
            var t = term.mkString("")
            positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t)))
            var np = positiveTot.count()

            if (np>0)
              positive = positiveTot

            if (useCache)
              positive = positive.cache()

            if (verbose){
              println("ptot: " + ptot)
              println("np: " + np)
            }

            //****Verificare l'effettiva utilità di questa istruzione*****
            if (ptot==np){
              println("break del while")
              break
            }

            ptot = np
            if (verbose){
              println("positivi rimasti: " + ptot)
              println("Funzione parziale ottenuta: " + f1)
            }

            //Rilegge le istanze negative
            negative = negativeTot
            if (useCache)
              negative = negative.cache()

            q = ntot
            p = ptot
            term = Array.fill[String] (n) ("*")
          } //end della if q<=0
        }

        if (verbose)
          println("Generated Function:" + f1 )

        if (printElapsed)
          println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

        return f1


      }
      /*******************************MR2_1 Elaborazione su istance negativr con tutte le positive in broadcast************************************************************************/
      /* Gestire le istanze positive come semplice array e non come DataFrame in mdoo da fare un solo take ed ottimizzare tutti i count.
          VA anche risctitta la funzione di eliminaPositivi
         Nella funzione Map rendere una variabile il denominatore in modo che sia invariante ad ogni giro
         Trasformare lo sparseVector in semplice Array

       */
       //Filtra i positivi superstiti nel calcolo degli Sij
       def checkPreserveInstanceArr(instances: Array[Array[Double]], top: Int, n: Int): Array[Array[Double]] = {
         var indiceTop=0
         var gamma=0.0
         var survivors = Array.empty[Array[Double]]

         if (top>n-1){
             indiceTop=top-n
             //gamma=false #Inutile perché la variabile è già inizializzata a false
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


       //Restituisce le istanze positive superstiti
       def filterPositiveInstances(instances: Array[Array[Double]], t: String): Array[Array[Double]] ={
         var survivors = Array.empty[Array[Double]]
         var term:Array[String] = t.split("")

         var posIter =instances.iterator
         while (posIter.hasNext)
         {
           var instance = posIter.next()
           breakable{
             for (i <-instance.indices){
                 if ((instance(i)==1.0 && term(i)=="0") || (instance(i)==0.0 && term(i)=="1")) {
                   survivors:+= instance
                   break
                   }
             }
          }
         }
         return survivors
       }

          def maxRilevanzaMap2_1(neg: org.apache.spark.sql.Row, pos:Array[Array[Double]], p: Long, q: Long, n: Int): org.apache.spark.ml.linalg.SparseVector  = {
              var negative = neg.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]  //Tipicamente sarà l'instanza negativa che dovrebbe avere più¢ardinalità

              var posIter =pos.iterator
              var v = Array.fill[Double] (n*2) (0.0)

              while (posIter.hasNext)
              {
                var positive = posIter.next()
                var posIndices = positive.indices
                var den=0.0
                //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
                //Conta numero di differenze tra positivo e negativo
                for (k <- posIndices){
                    if (positive(k)!=negative(k)) den +=1
                  }
                var oneOnDen = 1/den
                //Calcola la rilevanza del k-esimo elemento della coppia i,j
                for (k <- posIndices){
                  if (positive(k)==1.0 && negative(k)==0.0)
                    v(k) +=  oneOnDen
                  else if (positive(k)==0.0 && negative(k)==1.0)
                    v(k+n) += oneOnDen
                }
              }
              return (new org.apache.spark.ml.linalg.DenseVector(v)).toSparse
            }

            def maxRilevanzaMR2_1(positives: Array[Array[Double]], negatives: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {
              var posList = sc.broadcast(positives)
              var rdd =negatives.rdd.map(row => maxRilevanzaMap2_1(row, posList.value,  p, q, n))
              var v:org.apache.spark.ml.linalg.SparseVector = rdd.reduce( (ril1,ril2) => maxRilevanzaRed(ril1,ril2,n))
              return v.argmax
            }

            def fromDFOfSparseVectorToArrayOfArrayDouble(df: DataFrame): Array[Array[Double]]={
              //Converte il Dataframe di Row Row di Sparsevector dei positivi  in Array di Array di Double (formato più leggero)
              var iter = df.takeAsList(df.count().asInstanceOf[Int]).iterator
              var arrOfarr = Array.empty[Array[Double]]
              while (iter.hasNext)
                arrOfarr:+= iter.next()(0).asInstanceOf[SparseVector].toArray
              return arrOfarr
            }

            def brainTrainingMR2_1(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true, sc: org.apache.spark.SparkContext): String = {
              val t0 = System.nanoTime()

              if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
              if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

              //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
              val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)

              var f1 = ""
              var funz1 = Array.empty[String]

              //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
              var n= positiveTotPar.head()(0).asInstanceOf[SparseVector].size
              println ("n: " + n)

              //numero totale dei positivi e negativi
              var ptot=positiveTotPar.count()
              var ntot=negativeTotPar.count()
              //numero dei positivi e negativi utilizzati per calcolare gli Si
              var p = ptot
              var q = ntot

              var negativeTot = negativeTotPar
              if (useCache)
                 negativeTot.cache()

              //Converte il Dataframe di Row Row di Sparsevector dei positivi  in Array di Array di Double (formato più leggero)
              var positiveTot = fromDFOfSparseVectorToArrayOfArrayDouble(positiveTotPar)

              var positive=positiveTot
              var negative=negativeTot

              if (verbose)
                 println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

              //Initizialize literal Function
              //var term = "*" * n
              var term = Array.fill[String] (n) ("*")

              //Indice dell'elemento a massima rilevanza per il set Sij
              var top=0

              while (ptot>0){ //sez. 2 Finché ci sono positivi
                //Calcola rilevanza massima
                top = maxRilevanzaMR2_1(positive, negative, p, q, n, sc); //2.3.1
                if (verbose) println ("Top: " + top)

                //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
                if (top > n - 1){
                    if (verbose) println("x^" + (top-n))
                    term(top - (n)) = "0"
                }else{
                    if (verbose) println("x" + top)
                    term(top) = "1"
                    }

                //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
                //positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
                positive = checkPreserveInstanceArr(positive, top, n)
                //if (useCache)
                  //positive = positive.cache()
                //p = positive.count()
                p = positive.size
                negative = negative.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))

                if (useCache)
                  negative.cache()

                var newQ = negative.count()
                if (q==newQ){
                  throw new Exception("Invalid data found in negative deletion! Training aborted.")
                }else{
                  q = newQ
                }
                if (verbose) println ("p = "+p+"; q = " + q)

                /*
                Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
                Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
                come fosse un repeat until utilizzando i negativi superstiti come indicatore
                */
                if (q <= 0){
                  //2.4 Add function term
                  funz1:+ term.mkString("")
                  if (f1.length()>0) f1 += " + "
                  for (i <- term.indices){
                      if (term(i) == "0")
                        f1 += "x"+(i+1)+"^"
                      else if (term(i) == "1")
                        f1 += "x"+(i+1)
                  }


                  //Elimina i termini positivi che soddisfano la funzione,
                  //o meglio seleziona solo i termini non soddisfatti
                  //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
                  println("ptot: " + ptot)
                  var t = term.mkString("")
                  //positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t))) //******RISCRIVERE ******
                  //println(t)
                  positiveTot = filterPositiveInstances(positiveTot,t)
                  //var np = positiveTot.count()
                  var np = positiveTot.size
                  if (np>0)
                    positive = positiveTot

                  //if (useCache)
                    //positive = positive.cache()

                  if (verbose){
                    println("ptot: " + ptot)
                    println("np: " + np)
                  }

                  //****Verificare l'effettiva utilità di questa istruzione*****
                  if (ptot==np){
                    println("break del while")
                    break
                  }

                  ptot = np
                  if (verbose){
                    println("positivi rimasti: " + ptot)
                    println("Funzione parziale ottenuta: " + f1)
                  }

                  //Rilegge le istanze negative
                  negative = negativeTot
                  if (useCache)
                     negative.cache()

                  q = ntot
                  p = ptot
                  term = Array.fill[String] (n) ("*")
                } //end della if q<=0
              }

              if (verbose)
                println("Generated Function:" + f1 )

              if (printElapsed)
                println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

              return f1
            }

            /*******************************MR2_2 - Sostituito il ritorno del MAp da SparseVector a Vector ************************************************************************/

            //Accetta in input e ritorna Array[Double] invede di SparseVector
            def maxRilevanzaRedMR2_2(x: Array[Double], y: Array[Double],n: Int): Array[Double] ={
              //Alternativa in una riga che però mi sembra più lenta:
              //(x, y).zipped.map(_ + _)
              var v = Array.fill[Double] (n*2) (0.0)
              for (k <- x.indices) {
                v(k) = x(k) + y(k)
              }
              return v
            }
            //Modificato per ritornare al reduce un Vettore e non uno SparseVector
            def maxRilevanzaMap2_2(neg: org.apache.spark.sql.Row, pos:Array[Array[Double]], p: Long, q: Long, n: Int): Array[Double]  = {
                var negative = neg.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]  //Tipicamente sarà l'instanza negativa che dovrebbe avere più¢ardinalità

                var posIter =pos.iterator
                var v = Array.fill[Double] (n*2) (0.0)

                while (posIter.hasNext)
                {
                  var positive = posIter.next()
                  var posIndices = positive.indices
                  var den=0.0
                  //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
                  //Conta numero di differenze tra positivo e negativo
                  for (k <- posIndices){
                      if (positive(k)!=negative(k)) den +=1
                    }
                  var oneOnDen = 1/den
                  //Calcola la rilevanza del k-esimo elemento della coppia i,j
                  for (k <- posIndices){
                    if (positive(k)==1.0 && negative(k)==0.0)
                      v(k) +=  oneOnDen
                    else if (positive(k)==0.0 && negative(k)==1.0)
                      v(k+n) += oneOnDen
                  }
                }
                return v
              }

              def maxRilevanzaMR2_2(positives: Array[Array[Double]], negatives: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {
                var posList = sc.broadcast(positives)
                var rdd =negatives.rdd.map(row => maxRilevanzaMap2_2(row, posList.value,  p, q, n))
                var v:Array[Double] = rdd.reduce( (ril1,ril2) => maxRilevanzaRedMR2_2(ril1,ril2,n))
                return v.indices.maxBy(v)
              }

              //def brainTrainingMR2_2(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true, sc: org.apache.spark.SparkContext=null): String = {
                def brainTrainingMR2_2(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true): String = {

                //Get the current Spark Context
                var sc = positiveTotPar.rdd.context
                val t0 = System.nanoTime()

                if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
                if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

                //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
                val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)

                var f1 = ""
                var funz1 = Array.empty[String]

                //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
                var n= positiveTotPar.head()(0).asInstanceOf[SparseVector].size
                println ("n: " + n)

                //numero totale dei positivi e negativi
                var ptot=positiveTotPar.count()
                var ntot=negativeTotPar.count()
                //numero dei positivi e negativi utilizzati per calcolare gli Si
                var p = ptot
                var q = ntot

                var negativeTot = negativeTotPar
                if (useCache)
                   negativeTot.cache()

                //Converte il Dataframe di Row Row di Sparsevector dei positivi  in Array di Array di Double (formato più leggero)
                var positiveTot = fromDFOfSparseVectorToArrayOfArrayDouble(positiveTotPar)

                var positive=positiveTot
                var negative=negativeTot

                if (verbose)
                   println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

                //Initizialize literal Function
                //var term = "*" * n
                var term = Array.fill[String] (n) ("*")

                //Indice dell'elemento a massima rilevanza per il set Sij
                var top=0

                while (ptot>0){ //sez. 2 Finché ci sono positivi
                  //Calcola rilevanza massima
                  top = maxRilevanzaMR2_2(positive, negative, p, q, n, sc); //2.3.1
                  if (verbose) println ("Top: " + top)

                  //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
                  if (top > n - 1){
                      if (verbose) println("x^" + (top-n))
                      term(top - (n)) = "0"
                  }else{
                      if (verbose) println("x" + top)
                      term(top) = "1"
                      }

                  //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
                  //positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
                  positive = checkPreserveInstanceArr(positive, top, n)
                  //if (useCache)
                    //positive = positive.cache()
                  //p = positive.count()
                  p = positive.size
                  negative = negative.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))

                  if (useCache)
                    negative.cache()

                  var newQ = negative.count()
                  if (q==newQ){
                    throw new Exception("Invalid data found in negative deletion! Training aborted.")
                  }else{
                    q = newQ
                  }
                  if (verbose) println ("p = "+p+"; q = " + q)

                  /*
                  Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
                  Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
                  come fosse un repeat until utilizzando i negativi superstiti come indicatore
                  */
                  if (q <= 0){
                    //2.4 Add function term
                    funz1:+ term.mkString("")
                    if (f1.length()>0) f1 += " + "
                    for (i <- term.indices){
                        if (term(i) == "0")
                          f1 += "x"+(i+1)+"^"
                        else if (term(i) == "1")
                          f1 += "x"+(i+1)
                    }


                    //Elimina i termini positivi che soddisfano la funzione,
                    //o meglio seleziona solo i termini non soddisfatti
                    //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
                    println("ptot: " + ptot)
                    var t = term.mkString("")
                    //positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t))) //******RISCRIVERE ******
                    //println(t)
                    positiveTot = filterPositiveInstances(positiveTot,t)
                    //var np = positiveTot.count()
                    var np = positiveTot.size
                    if (np>0)
                      positive = positiveTot

                    //if (useCache)
                      //positive = positive.cache()

                    if (verbose){
                      println("ptot: " + ptot)
                      println("np: " + np)
                    }

                    //****Verificare l'effettiva utilità di questa istruzione*****
                    if (ptot==np){
                      println("break del while")
                      break
                    }

                    ptot = np
                    if (verbose){
                      println("positivi rimasti: " + ptot)
                      println("Funzione parziale ottenuta: " + f1)
                    }

                    //Rilegge le istanze negative
                    negative = negativeTot
                    if (useCache)
                       negative.cache()

                    q = ntot
                    p = ptot
                    term = Array.fill[String] (n) ("*")
                  } //end della if q<=0
                }

                if (verbose)
                  println("Generated Function:" + f1 )

                if (printElapsed)
                  println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

                return f1
              }
              /*******************************MR3 - Elaborazione su istance postiive con tutte le negative in broadcast (Ottimizzato per usare Array[Double] invece di SParseVector) ************************************************************************/

                            def maxRilevanzaMap3(pos: org.apache.spark.sql.Row, neg:java.util.List[org.apache.spark.sql.Row], p: Long, q: Long, n: Int): org.apache.spark.ml.linalg.SparseVector  = {
                                var positive = pos.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]  //Tipicamente sarà l'instanza negativa che dovrebbe avere più¢ardinalità

                                var negIter =neg.iterator
                                var v = Array.fill[Double] (n*2) (0.0)

                                while (negIter.hasNext)
                                {
                                  var negative = negIter.next()(0).asInstanceOf[SparseVector].toArray
                                  var negIndices = negative.indices
                                  var den=0.0
                                  //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
                                  //Conta numero di differenze tra positivo e negativo
                                  for (k <- negIndices){
                                      if (positive(k)!=negative(k)) den +=1
                                    }
                                  //Calcola la rilevanza del k-esimo elemento della coppia i,j
                                  for (k <- negIndices){
                                    //println("k = " + k + " ((1 / den) / q) / p=" + (((1 / den) / q) / p) )
                                    if (positive(k)==1.0 && negative(k)==0.0)
                                      //v(k) +=  ((1 / den) / q) / p  //Attenzione ho reso implicito il cast a float. Verificare se è ok
                                      v(k) +=  (1 / den)
                                    else if (positive(k)==0.0 && negative(k)==1.0)
                                      //v(k+n) += ((1 / den) / q) / p //Attenzione ho reso implicito il cast a float. Verificare se è ok
                                      v(k+n) += (1 / den)
                                  }
                                }
                                return (new org.apache.spark.ml.linalg.DenseVector(v)).toSparse
                              }

                              def maxRilevanzaMR3(positives: org.apache.spark.sql.DataFrame, negatives: org.apache.spark.sql.DataFrame, p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {

                                var negList = sc.broadcast(negatives.takeAsList(p.asInstanceOf[Int]))
                                //var posList = positives.takeAsList(p.asInstanceOf[Int])
                                var rdd =positives.rdd.map(row => maxRilevanzaMap3(row, negList.value,  p, q, n))
                                //var rdd =negatives.rdd.map(row => maxRilevanzaMap2(row, posList,  p, q, n))
                                var v:org.apache.spark.ml.linalg.SparseVector = rdd.reduce( (ril1,ril2) => maxRilevanzaRed(ril1,ril2,n))
                                return v.argmax //Ritorna posizione del massimale. Verificare che vada bene come massimo dell'algoritmo **********
                                //v.toArray.indexOf(v.toArray.max) //Ottimizzare
                              }

                              def brainTrainingMR3(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true, sc: org.apache.spark.SparkContext): String = {
                                val t0 = System.nanoTime()

                              //  var positiveTotPar = positiveTotParDs.toDF()
                                var negativeTot = negativeTotPar

                                var positiveTot = positiveTotPar

                                if (positiveTot.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
                                if (negativeTot.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

                                //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
                                val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)
                                val checkRetainPositiveUDF = udf(checkRetainPositiveInstances(_: SparseVector, _: String ): Boolean)

                                var f1 = ""
                                var funz1 = Array.empty[String]

                                if (useCache){
                                  positiveTot = positiveTot.cache()
                                  negativeTot = negativeTot.cache()
                                }
                                //***Verificare che effettivamente avvenga la copia
                                var positive=positiveTot
                                var negative=negativeTot

                                //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
                                var n= positiveTot.head()(0).asInstanceOf[SparseVector].size
                                println ("n: " + n)

                                //numero totale dei positivi e negativi
                                var ptot=positiveTot.count()
                                var ntot=negativeTot.count()
                                //numero dei positivi e negativi utilizzati per calcolare gli Si
                                var p = ptot
                                var q = ntot

                                if (verbose)
                                   println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

                                //Initizialize literal Function
                                //var term = "*" * n
                                var term = Array.fill[String] (n) ("*")

                                //Indice dell'elemento a massima rilevanza per il set Sij
                                var top=0

                                while (ptot>0){ //sez. 2 Finché ci sono positivi
                                  //Calcola rilevanza massima
                                  top = maxRilevanzaMR2(positive, negative, p, q, n, sc); //2.3.1
                                  if (verbose) println ("Top: " + top)

                                  //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
                                  if (top > n - 1){
                                      if (verbose) println("x^" + (top-n))
                                      term(top - (n)) = "0"
                                  }else{
                                      if (verbose) println("x" + top)
                                      term(top) = "1"
                                      }

                                  //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
                                  positive = positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
                                  if (useCache)
                                    positive = positive.cache()
                                  p = positive.count()
                                  negative = negative.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
                                  if (useCache)
                                    negative = negative.cache()
                                  var newQ = negative.count()
                                  if (q==newQ){
                                    throw new Exception("Invalid data found in negative deletion! Training aborted.")
                                  }else{
                                    q = newQ
                                  }
                                  if (verbose) println ("p = "+p+"; q = " + q)

                                  /*
                                  Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
                                  Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
                                  come fosse un repeat until utilizzando i negativi superstiti come indicatore
                                  */
                                  if (q <= 0){
                                    //2.4 Add function term
                                    funz1:+ term.mkString("")
                                    if (f1.length()>0) f1 += " + "
                                    for (i <- term.indices){
                                        if (term(i) == "0")
                                          f1 += "x"+(i+1)+"^"
                                        else if (term(i) == "1")
                                          f1 += "x"+(i+1)
                                    }


                                    //Elimina i termini positivi che soddisfano la funzione,
                                    //o meglio seleziona solo i termini non soddisfatti
                                    //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
                                    println("ptot: " + ptot)
                                    var t = term.mkString("")
                                    positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t)))
                                    var np = positiveTot.count()

                                    if (np>0)
                                      positive = positiveTot

                                    if (useCache)
                                      positive = positive.cache()

                                    if (verbose){
                                      println("ptot: " + ptot)
                                      println("np: " + np)
                                    }

                                    //****Verificare l'effettiva utilità di questa istruzione*****
                                    if (ptot==np){
                                      println("break del while")
                                      break
                                    }

                                    ptot = np
                                    if (verbose){
                                      println("positivi rimasti: " + ptot)
                                      println("Funzione parziale ottenuta: " + f1)
                                    }

                                    //Rilegge le istanze negative
                                    negative = negativeTot
                                    if (useCache)
                                      negative = negative.cache()

                                    q = ntot
                                    p = ptot
                                    term = Array.fill[String] (n) ("*")
                                  } //end della if q<=0
                                }

                                if (verbose)
                                  println("Generated Function:" + f1 )

                                if (printElapsed)
                                  println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

                                return f1
                              }

      /*******************************MR3_1 - Elaborazione su istance postiive con tutte le negative in broadcast************************************************************************/


      //Accetta in input e ritorna Array[Double] invede di SparseVector
      def maxRilevanzaRedMR3_1(x: Array[Double], y: Array[Double],n: Int): Array[Double] ={
        var v = Array.fill[Double] (n*2) (0.0)
        for (k <- x.indices) {
          v(k) = x(k) + y(k)
        }
        return v
      }


      //Modificato per ritornare al reduce un Vettore e non uno SparseVector
      def maxRilevanzaMap3_1(pos: org.apache.spark.sql.Row, neg:Array[Array[Double]], p: Long, q: Long, n: Int): Array[Double]  = {
          var positive = pos.getAs("features").asInstanceOf[org.apache.spark.ml.linalg.SparseVector]  //Tipicamente sarà l'instanza negativa che dovrebbe avere più¢ardinalità

          var negIter =neg.iterator
          var v = Array.fill[Double] (n*2) (0.0)

          while (negIter.hasNext)
          {
            var negative = negIter.next()
            var negIndices = negative.indices
            var den=0.0
            //<todo IMPORTANTE!> Questo valore può essere calcolato una volta per tutte per tutto il dataframe****
            //Conta numero di differenze tra positivo e negativo
            for (k <- negIndices){
                if (positive(k)!=negative(k)) den +=1
              }
            var oneOnDen = 1/den
            //Calcola la rilevanza del k-esimo elemento della coppia i,j
            for (k <- negIndices){
              if (positive(k)==1.0 && negative(k)==0.0)
                v(k) +=  oneOnDen
              else if (positive(k)==0.0 && negative(k)==1.0)
                v(k+n) += oneOnDen
            }
          }
          return v
        }

        def maxRilevanzaMR3_1(positives: org.apache.spark.sql.DataFrame, negatives: Array[Array[Double]], p: Long, q: Long, n: Int,sc: org.apache.spark.SparkContext): Int = {
          var negList = sc.broadcast(negatives)
          var rdd =positives.rdd.map(row => maxRilevanzaMap3_1(row, negList.value,  p, q, n))
          var v:Array[Double] = rdd.reduce( (ril1,ril2) => maxRilevanzaRedMR3_1(ril1,ril2,n))
          return v.indices.maxBy(v)
        }

        def brainTrainingMR3_1(positiveTotPar: DataFrame , negativeTotPar: DataFrame, verbose: Boolean = true, printElapsed: Boolean = true, useCache: Boolean = true): String = {
          //Get the current Spark Context
          var sc = positiveTotPar.rdd.context

          val t0 = System.nanoTime()

          if (positiveTotPar.head(1).isEmpty) throw new IllegalArgumentException("Positive instances Dataframe is empty!")
          if (negativeTotPar.head(1).isEmpty) throw new IllegalArgumentException("Negative instances Dataframe is empty!")

          //Registra le UDF - User Difined Functions in modo da poter essere invocate direttamente nell'engine di Spark
          val checkPreserveInstanceUDF = udf(checkPreserveInstance(_: SparseVector, _: Int, _: Int): Boolean)
          val checkRetainPositiveUDF = udf(checkRetainPositiveInstances(_: SparseVector, _: String ): Boolean)

          var f1 = ""
          var funz1 = Array.empty[String]

          //Stabilisce la lunghezza della sequenza (riga), ossia il numero delle features
          var n= positiveTotPar.head()(0).asInstanceOf[SparseVector].size
          println ("n: " + n)

          //numero totale dei positivi e negativi
          var ptot=positiveTotPar.count()
          var ntot=negativeTotPar.count()
          //numero dei positivi e negativi utilizzati per calcolare gli Si
          var p = ptot
          var q = ntot

          var positiveTot = positiveTotPar

          if (useCache)
             positiveTot.cache()

          //Converte il Dataframe di Row Row di Sparsevector dei positivi  in Array di Array di Double (formato più leggero)
          var negativeTot = fromDFOfSparseVectorToArrayOfArrayDouble(negativeTotPar)

          var positive=positiveTot
          var negative=negativeTot

          if (verbose)
             println("Creo funzione per classe 1 con p=" + p + "; q = " + q)

          //Initizialize literal Function
          //var term = "*" * n
          var term = Array.fill[String] (n) ("*")

          //Indice dell'elemento a massima rilevanza per il set Sij
          var top=0

          while (ptot>0){ //sez. 2 Finché ci sono positivi
            //Calcola rilevanza massima
            top = maxRilevanzaMR3_1(positive, negative, p, q, n, sc); //2.3.1
            if (verbose) println ("Top: " + top)

            //2.3.2 Add variable: Select the variable vk such that R(vk) is maximum. m= m + vk
            if (top > n - 1){
                if (verbose) println("x^" + (top-n))
                term(top - (n)) = "0"
            }else{
                if (verbose) println("x" + top)
                term(top) = "1"
                }

            //2.3.3 Erase positive and negative instances in order to erase Si not invcluding Vk and Sij including Vk
            //positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))
            negative = checkPreserveInstanceArr(negative, top, n)
            //if (useCache)
              //positive = positive.cache()
            //p = positive.count()
            //p = positive.count()
            positive = positive.filter(checkPreserveInstanceUDF(col("features"), lit(top), lit(n)))

            if (useCache)
              positive.cache()

            var newQ = negative.size
            if (q==newQ){
              throw new Exception("Invalid data found in negative deletion! Training aborted.")
            }else{
              q = newQ
            }
            if (verbose) println ("p = "+p+"; q = " + q)

            /*
            Se le istanze negative sono finite allora il ciclo è terminato e passo al prossimo positivo
            Di fatto questo controllo è equivalente a quello del While fatto nell'algoritmo. Ma viene fatto
            come fosse un repeat until utilizzando i negativi superstiti come indicatore
            */
            if (q <= 0){
              //2.4 Add function term
              funz1:+ term.mkString("")
              if (f1.length()>0) f1 += " + "
              for (i <- term.indices){
                  if (term(i) == "0")
                    f1 += "x"+(i+1)+"^"
                  else if (term(i) == "1")
                    f1 += "x"+(i+1)
              }


              //Elimina i termini positivi che soddisfano la funzione,
              //o meglio seleziona solo i termini non soddisfatti
              //***Questa parte va ottimizzata innanzitutto perché la stringa è convertita due volte e poi perché usare una stringa è poco efficiente******
              println("ptot: " + ptot)
              var t = term.mkString("")
              positiveTot = positiveTot.filter(checkRetainPositiveUDF(col("features"),lit(t)))

              var np = positiveTot.count()

              if (np>0)
                positive = positiveTot

              if (useCache)
                  positive = positive.cache()


              if (verbose){
                println("ptot: " + ptot)
                println("np: " + np)
              }

              //****Verificare l'effettiva utilità di questa istruzione*****
              if (ptot==np){
                println("break del while")
                break
              }

              ptot = np
              if (verbose){
                println("positivi rimasti: " + ptot)
                println("Funzione parziale ottenuta: " + f1)
              }

              //Rilegge le istanze negative
              negative = negativeTot
              //if (useCache)
              //   positive.cache()

              q = ntot
              p = ptot
              term = Array.fill[String] (n) ("*")
            } //end della if q<=0
          }

          if (verbose)
            println("Generated Function:" + f1 )

          if (printElapsed)
            println("Training time: " + (System.nanoTime() - t0)/ 1000000000.0 )

          return f1
        }

      def main22222(args: Array[String]): Unit = {
          if (args.length<2){
            println("Please provide following parameters: brain <algorithm> <training-set-path> <num-partitions-pos> <num-partitions-neg> ")
            println("Where algotithm in: MR2_2 (positivi in broadcast), MR3_1 (negativi in broadcast), MR (Cross), V (Cross uno per giro) ")
            System.exit(1)
          }

          val spark = SparkSession.builder().appName("Spark-Brain").getOrCreate()
          var instances = spark.read.parquet(args(1))
          var positive = instances.filter("label=1").select("features")
          var negative = instances.filter("label=0").select("features")

          if (args.length>=3 && args(3).toInt>0 )
          {
            positive = positive.repartition(args(2).toInt)
          }
          if (args.length>=4 && args(3).toInt>0)
            negative = negative.repartition(args(3).toInt)

          positive.collect()
          negative.collect()
          val t0 = System.nanoTime()
          var f=""
          args(0) match {
            case "V" => {
              f = brain.scala.BrainScala.brainTrainingV(positive, negative,false,false,true)
            }
            case "MR" => {
              f = brain.scala.BrainScala.brainTrainingMR(positive, negative,false,false,true)
            }
            case "MR2_2" => {
              f = brain.scala.BrainScala.brainTrainingMR2_2(positive, negative,false,false,true)
            }
            case "MR3_1" => {
              f = brain.scala.BrainScala.brainTrainingMR3_1(positive, negative,false,false,true)
            }

          }

          val trainingTime = (System.nanoTime() - t0)/ 1000000000.0
          println("Brain Version: "+args(0))
          println("Positive partitions: " + positive.rdd.getNumPartitions)
          println("Negative partitions: " + negative.rdd.getNumPartitions)
          println("f: "+f)
          println("Training Time: "+trainingTime)
        }


    }
