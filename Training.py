#Import Spark 
import findspark
findspark.init()
findspark.find()

#Import Libraries: Pyspark, Numpy, Pandas, sklearn
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Begin Spark Session
conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
spark_conf = pyspark.SparkContext(conf=conf)
spark_session = SparkSession(spark_conf)

#Load the TrainingDataset
data_frame = spark_session.read.format("csv").load("training_files/TrainingDataset.csv" , header = True ,sep =";")
data_frame.printSchema()
data_frame.show()

#Rename column 'quality' to column 'label'
for col_name in data_frame.columns[1:-1]+['""""quality"""""']:
    data_frame = data_frame.withColumn(col_name, col(col_name).cast('float'))
data_frame = data_frame.withColumnRenamed('""""quality"""""', "label")

features =np.array(data_frame.select(data_frame.columns[1:-1]).collect())
label = np.array(data_frame.select('label').collect())

VectorAssembler = VectorAssembler(inputCols = data_frame.columns[1:-1] , outputCol = 'features')
data_frame_tr = VectorAssembler.transform(data_frame)
data_frame_tr = data_frame_tr.select(['features','label'])

def lbl_points(spark_conf, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return spark_conf.parallelize(labeled_points) 

dataset = lbl_points(spark_conf, features, label)
training, test = dataset.randomSplit([0.7, 0.3],seed =11)

#Using random forest training classifier
RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)

predictions = RFmodel.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
labelsAndPredictions_df = labelsAndPredictions.toDF()
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()

#Calculate the F1score
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1-Score: ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy" , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

#save training model
RFmodel.save(spark_conf, 'output/trainingmodel.model')