# SPBRAIN a Big Data implementation using Apache Spark of a relevance-based learning algorithm 

## Introduction
SPBRAIN is a brand-new iplementation of U-BRAIN, a supervised machine learning algorithm inferring a binary classification function from the data input given. Initially conceived for the splicing site prediction problem, the algorithm is characterized by a great versatility and effectiveness, highlighted in numerous applications. However, the memory space and the execution time required, respectively of order O(n3) and O(n5), appear unacceptable for huge data sets computing. Thus, a new set of implementations based on big data paradigms and technologies, in particular Map-Reduce, have been implemented.

To test the performance of the new implementation, the problem of Splicing Site Prediction is addressed.  This is still a very important bioinformatics problem whose purpose has shifted from the identification of possible exon-intron boundaries before the Human Genome Project (HGP) was completed in 2003 to the prediction of the transcriptional impact of mutations at known splice sites and their vicinity regions in the post-HGP era.

In order to provide an effective, efficient and reliable (this aspect is very important in case of very long running executions) implementation, Apache Spark, one of the most interesting and used technologies in the big data field, available with open source license and available in the cloud computing facilities of the main world players, has been selected. The implementation benefits of the whole Hadoop ecosystem’s components, such as HDFS distributed file system, Yarn scheduler, Kafka streaming and many others more. The implementation is exploitable in numerous contexts from stand-alone applications up to streaming contexts, both in local and cloud environments.

## Algotirhms description
SP-BRAIN consists of three different implementation approaches: Cross-Join, Broadcast Positive and Broadcast Negative and a common framework. All implementation generates the same classification function, with different training times due of different implementation.
The best performer and more stable implementation (using the provided datasets) is the Broadcast Positive.


## Datasets
The test datasets used are IPDATA (Irvine Primate splice-junction data set), an two subsets of HS3D (Homo Sapiens Splice Sites Dataset).
IPDATA is a data set of human splice sites, and it consists of 767 donor splice sites, 765 acceptor splice sites, and 1654 false splice sites. Here have been considered 464 positive instances and 1536 negatives.
HS3D is a data set of Homo Sapiens Exon, Intron and Splice sites extracted from GenBank. It includes 2796 + 2880 donor and acceptor sites, as windows of 140 nucleotides around a splice site, and 271,937+332,296 windows of false splice sites, selected by searching canonical GT-AG pairs in not splicing positions.
The datasets information are resumed in the following table.

| Dataset  | #Nucleotides | Training Inst. (pos./neg.) | Test.  Instances (pos./neg.) | Total samples |
| ---------| -------------|----------------------------|------------------------------|---------------|
|IPDATA	   |60	          |464/1536	                   |302/884	                      |     3186      |
|HS3D_UB	 | 140	        |161/2794	                   |69/1197	                      |     4221      |
|HS3D_2	   | 140	        |1960/12571	                 |836/5431	                    |     20768     |


## Content of folder
* datasets: datasets for splicing site prediction HS3D and IPDATA (txt and parquet format)
* src: SP-BRAIN sources
* bin: SP-BRAIN binary package in jar format 
* python-exaples: Python example using SP-BRAIN

----------------------------------------------------------------------------------------------------------------------------------------
## How to Run Code

### Run code manually usign spark-shell
From command line execute: spark-shell –jars <path-of-jar>/spBRAIN1_0.jar  --driver-memory 4g –master local[*]
```
var instances = spark.read.parquet(“BRAINscala/ipdata/ipdata-training-parquet”)
var negative = instances.filter(“label=0”).select(“features”)
var positive = instances.filter(“label=1”).select(“features”)
val trainedModel = BRAIN.scala.BRAINScalaMRCROSS.fit(positive.coalesce(1), negative.repartition(4), false,true)
val test = spark.read.parquet(“BRAINscala/ipdata/ipdata-test-parquet”)
val model = brain.scala.BrainScalaModel.load(“ipdata”)
val calssified = model.transform(test)
classified.show()
```

### Run code using Hadoop cluster via spark-submit
##### Run application locally on 8 cores
```
./bin/spark-submit \
  --class brain.scala.BrainScalaMRBPTester \
  --master local[8] \
  brain-scala-utils_2.11-1.0.jar \
 ipdata/ipdata-training-parquet     ipdata/models/ipdata-model-A        1       8
```

##### Run on a Spark standalone cluster in client deploy mode using 32 cores
```
./bin/spark-submit \
  --class brain.scala.BrainScalaMRBPTester \
  --master spark://207.184.161.138:7077 \
  --executor-memory 20G \
  --total-executor-cores 64 \
  brain-scala-utils_2.11-1.0.jar \
 ipdata/ipdata-training-parquet       ipdata/models/ipdata-model-B      1      32
```
  
#### Run code on a YARN cluster using 64 cores
```
./bin/spark-submit \
--class brain.scala.BrainScalaMRBPTester \
  --master yarn \
  --deploy-mode cluster \  
  --executor-memory 40G \
  --num-executors 64 \
  brain-scala-utils_2.11-1.0.jar \
 ipdata/ipdata-training-parquet    ipdata/models/ipdata-model-C   1      64
```

### Run using Jupyter Notebook
Please refer to file: python-examples/SP-BRAINPythonTester.ipynb

## Run using Cloud environments
The job can be sumitted using a cloud ready-to-use environment.
Here the parameter for GCP – Google Cloud Platform, that provides a web GUIs to submit and monitoring spark jobs without accessing to a command shell are often available, are reported.
To access the platform, the creation of an account is needed.
Using the web GUI it is simple to create a cluster.
Datasets can be update to a Bucket cloud storage or local disk, than a Spark job can be submitted via GUI providing the following parameters:
* Job Id: an arbitraty name used to find the job in the Yarn GUI
* Region: Region of your cluster (a cloud parameter)
* Cluster: the name of the cluster to execute the job
* Job type: Spark
* Main class or Jar: brain.scala.BrainScalaMRBPTester
* Arguments:
** Training dataset in parquet format path (if on Bucket cloud storage is domething like: gs://dataproc..../h23d2-training-parquet)
** Output function path/filename (bucket od local)
** Number of positive partitions to be used (e.g. 1)
** Number of negative partitions to be used (e.g. 64) 
* Jar files: <path-to-jar>/brain-scala-utils_2.11-1.0.jar
  
----------------------------------------------------------------------------------------------------------------------------------------
## API user manual
Please refer to [User manual](user-manual.md)

## Author
You can get in touch with me on my LinkedIn Profile: [![LinkedIn Link](https://img.shields.io/badge/Connect-valerio75-blue.svg?logo=linkedin&longCache=true&style=social&label=Connect)](https://www.linkedin.com/in/valerio-morfino/)


## Issues
If you face any issue, you can create a new issue in the Issues Tab and I will be glad to help you out.
[![GitHub Issues](https://img.shields.io/github/issues/valerio75/spbrain.svg?style=flat&label=Issues&maxAge=2592000)](https://github.com/valerio75/spbrain/issues)


## License
[GNU General Public License v3.0](LICENSE)

Copyright (c) 2018-present, Valerio Morfino                                                        

