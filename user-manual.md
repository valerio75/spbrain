The features offered by SP-BRAIN are of four kind of category: Training features, Transform features, Exploring inferred module features, Load/Save model.
The full Scaladoc is provided together with the source code.

# 1.	Preliminary operations
Prerequisite to use SP-BRAIN is to have a working Spark environment at least v. 2.1.
In order to work with SP-BRAIN, regardless of the working context chosen, is to import the brainscala.jar file in the execution context. In the following paragraphs some examples in different contexts are described.

# 2.	Training a model
To train a model it is necessary to choose a version of SP-BRAIN. Three different versions are currently available: BrainScalaMRBP, BrainScalaMRBN, BrainScalaMRCROSS. For each version a class is provided. 
Regardless of the version, it is necessary to call the "fit" method to perform the training operation:

```
fit(positive: DataFrame, negative: DataFrame, verbose: Boolean = false, useCache: Boolean = true): BrainScalaModel
```

The method returns a BrainScalaModel, an object representing a trained model.
The parameters to pass to the fit method are:
*	positive: org.apache.spark.sql.DataFrame, containing positive instances without labels, encoded as a SparseVector of Double. The encoding is explained in paragraph 2.3.
*	negative: org.apache.spark.sql.DataFrame. containing negative instances without labels, encoded as a SparseVector of Double, using same encoding of positive instances
*	verbose: Boolean, enable verbose mode. Default value false
*	useCache: Boolean, enable use of caching. Default value true

# 3.	Transform unclassified data
The class BrainScalaModel offers the method transform to classify unlabelled data
```
transform(instances: DataFrame):DataFrame 
```
The method accept a org.apache.spark.sql.DataFrame containing a SparseVector of Double containing unlabelled instances and returns a new DataFrame containing, in addition to the columns supplied as input, a new class column, which contains the predicted label.
To use this method it is necessary to have an instance of BrainScalaModel, by training a model or loading a persisted one.

# 4. Exploring a trained model
A model inferred by fit function or loaded can be explored accessing to the following attributes:
*	funz: Array[Array[Double]], trained model as Array
*	funzTxt: String, trained model as string
*	trainingNeg: Long, count of negative instances used for training the model
*	trainingPos: Long, count of positive instances used for training the model
*	trainingTime: Double, model training time
*	version: String, version of SP-BRAIN used to train the model

# 5.	Saving a model
BrainScalaModel has the method:
```
saveModel(destination:String)
```
This method, that is not static, so needs a trainedObject, accept as parameter a string containing the URI of destination. The destination can be a local file system or HDFS. The save operation creates two directory in parquet format for each model saved. 
Given the name “myTrainedModel” to the saveModel method, the following directories are generated:
*	myTrainedModel.metadata
*	myTrainedModel.model

# 6.	Loading a model
A previously saved trained model can be loaded using the static method loadModel of the class BrainScalaModel:
loadModel(filePath:String):BrainScalaModel
The method accepts as parameter the path (local or HDFS) of a previously saved model and returns an instance of BrainScalaModel.
Note that the name to provide is the model name without “metadata” or “model” extensions.

# 7.	Tester classes
All algorithm’s versions have a Tester class that is useful when a Job have to be submitted or, however, in similar cases.
Each Test class has a main method and execute the following tasks:
1.	Load training data from a Parquet file
2.	Repartition data
3.	Train a model
4.	Save the inferred model

The program accepts the following parameters on the command line:
*	Full path of training dataset to be used, in Parquet format
*	Full path of the model inferred to be saved
*	Number of partition of positive instances
*	Number of partition of negative instances

A Test command-line invocation template is:
```
brain.scala.<algorithm-version>Tester  <training-data-path> <inferred-model-save-path> <positive-partitions-number> <negative-partitions-number>
```
