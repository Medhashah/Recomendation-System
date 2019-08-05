import sys

# And pyspark.sql to get the spark session
from pyspark import SparkContext
from pyspark.sql import SparkSession


from pyspark import SparkConf
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.feature import StringIndexer
from pyspark.ml import PipelineModel
from pyspark.sql.functions import *
from pyspark.sql.functions import log

# TODO: you may need to add imports here


def main(spark, model_file, data_file):
    '''Main routine for supervised evaluation
    Parameters
    ----------
    spark : SparkSession object
    model_file : string, path to store the serialized model file
    data_file : string, path to the parquet file to load
    
    '''
    test_data = spark.read.parquet(data_file)
    test_data= test_data.withColumn("logcount", log(test_data["count"]))


    model = PipelineModel.load(model_file)

    pred = model.transform(test_data)

    userRecs = model.stages[-1].recommendForAllUsers(500)

    userRecs = userRecs.select("user","recommendations.track")

    #testData = test_data.rdd.map(lambda r: r.user , r.track )
    testData = pred.select('user','track')

    groundTruth = testData.rdd.groupByKey().mapValues(list).toDF()

    scoreAndLabels = userRecs.join(groundTruth , col("user") == col("_1") ).rdd.map(lambda tup: (tup[1],tup[3]))

    metrics = RankingMetrics(scoreAndLabels)

    print("RMSE with RankingMetrics = " + str(metrics.meanAveragePrecision))




    '''
    testData = test_data.rdd.map(lambda p: (p.user, p.track))
    #predictions = model.recommendForAllUsers(testData).rdd.map(lambda r: ((r.user, r.track), r.count))
    predictions = model.predictAll(testData).map(lambda r: ((r.user, r.product), r.rating))
    userRecs = model.recommendForAllUsers(10)
    userRecs.show(10)
    ratingsTuple = test_data.rdd.map(lambda r: ((r.user, r.track), r.count))
    testData = test_data.rdd.map(lambda r: r.user , r.track )
    temp2 = testData.rdd.groupByKey().mapValues(list).toDF()
    ratingsTuple.collect()
    scoreAndLabels = userRecs.rdd.join(ratingsTuple).map(lambda tup: tup[0])
    scoreAndLabels.collect()
    metrics = RankingMetrics(scoreAndLabels)
    print(metrics.precisionAt(10))
    score = userRecs.join(temp2 , col("user") == col("_1") )
    scoreAndLabels = userRecs.join(temp2 , col("user") == col("_1") ).rdd.map(lambda tup: (tup[1],tup[2]))
    metrics = RankingMetrics(scoreAndLabels)
    '''


        
   

    ###
    # TODO: YOUR CODE GOES HERE
    ###

    pass




# Only enter this block if we're in main
if __name__ == "__main__":

    sc_conf = SparkConf()
    sc_conf.set('spark.executor.memory', '8g')
    sc_conf.set('spark.driver.memory', '8g')

    # Create the spark session object
    spark = SparkSession.builder.appName('test').config(conf=sc_conf).getOrCreate()

    # And the location to store the trained model
    model_file = sys.argv[1]

    # Get the filename from the command line
    data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, data_file)
