import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark import SparkConf

# TODO: you may need to add imports here
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline




def main(spark, data_file, val_file, model_file):
    '''Main routine for unsupervised training
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''


    train_data = spark.read.parquet(data_file)
    train_data = train_data.sample(False,0.1)
    val_data = spark.read.parquet(val_file)
    val_data = val_data.sample(False , 0.1)

    train_data.createOrReplaceTempView('train_data')
    #train_data.show()
    train_data1=spark.sql('select * from train_data WHERE count < 20')
    indexer_user_id = StringIndexer(inputCol="user_id", outputCol="user", handleInvalid ="skip")
#     train_data1 = indexer_user_id.fit(train_data1).transform(train_data1)
    indexer_track_id = StringIndexer(inputCol="track_id", outputCol="track", handleInvalid ="skip")
#     train_data1 = indexer_track_id.fit(train_data1).transform(train_data1)
   # train_data1.show()

    ranks =[ 4]
    alphas = [0.3]
    regularizations = [0.01]


    train_data = train_data.cache()
    val_data = val_data.cache()
    bestModel = None
    ranks =[ 4]
    alphas = [0.3]
    regularizations = [0.01]


    train_data = train_data.cache()
    val_data = val_data.cache()
    bestModel = None

    best_rmse = None

    best_rank = None
    best_alpha = None
    best_regularization = None

    for r in ranks :
        for a in alphas :
            for rp in regularizations :

                als = ALS(maxIter = 5 , regParam= rp, userCol= "user" , itemCol= "track" , ratingCol ="count" , implicitPrefs=True , coldStartStrategy="drop" , alpha = a , rank = r)

                pipeline = Pipeline(stages=[indexer_user_id, indexer_track_id, als])
                
                model = pipeline.fit(train_data)
                
                predictions = model.transform(val_data)
                
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
                
                rmse = evaluator.evaluate(predictions)

                if best_rmse is None or best_rmse > rmse :
                    best_rmse = rmse
                    bestModel = model
                    best_rank = r
                    best_alpha = a
                    best_regularization = rp


    print("The best hyper parameters(count >2) are as follows:")
    print("rank: " + str(best_rank))
    print("alpha: " + str(best_alpha))
    print("regularParam: " + str(best_regularization))
    print("And with root mean square error:" + str(best_rmse))


    bestModel.save(model_file)
    pass
if __name__ == "__main__":

    sc_conf = SparkConf()
    sc_conf.set('spark.executor.memory', '8g')
    sc_conf.set('spark.driver.memory', '8g')

    # Create the spark session object
    spark = SparkSession.builder.appName('train').config(conf=sc_conf).getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # Get the filename from the command line
    val_file = sys.argv[2]

    # And the location to store the trained model
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, data_file, val_file, model_file)
  
