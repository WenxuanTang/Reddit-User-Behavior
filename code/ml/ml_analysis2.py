# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in data

# COMMAND ----------

dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet")

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter out the number of posts with untrackable authors<br>
# MAGIC #### (Might due to user cancellation or user-deletion of the post)

# COMMAND ----------

from pyspark.sql.functions import col

deleted = comments.filter(col("subreddit") == "AskReddit").filter(col("author") == "[deleted]")
print(deleted.count())
deleted.show(5)

# COMMAND ----------

askreddit = comments.filter(col("subreddit") == "AskReddit")
askreddit.write.mode('Overwrite').parquet('dbfs:/FileStore/comments_askreddit.parquet')

# COMMAND ----------

comment_ar = spark.read.parquet('dbfs:/FileStore/comments_askreddit.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter out the actual cancelled users, randomly sample similar number of trackable posts to balance the dataset

# COMMAND ----------

deleted_user = comment_ar.filter((col("author") == "[deleted]") & (col("body") != "[deleted]"))
print(deleted_user.count())
not_deleted = comment_ar.filter(col("body") != "[deleted]").sample(fraction = 6893622/(comment_ar.count()-6893622), seed = 685)
print(not_deleted.count())

# COMMAND ----------

sampled = deleted_user.union(not_deleted)
sampled.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove useless columns, add feature for removed-by-mod posts, then separate them temporarily
# MAGIC #### (Since their bodies are plainly [removed], useless for sentiment analysis and keyword extraction)

# COMMAND ----------

from pyspark.sql.functions import *

sampled_new = sampled.drop('author_cakeday','author_flair_css_class','author_flair_text','can_gild','distinguished','retrieved_on','id','link_id','parent_id','permalink','subreddit','subreddit_id').dropna()
sampled_new = sampled_new.withColumn("removed_by_mod", col("body") == "[removed]")
sampled_new_2021 = sampled_new.filter(year(from_unixtime(col('created_utc'))) == 2021)
sampled_not_removed = sampled_new_2021.filter(col("removed_by_mod") == False)
sampled_removed = sampled_new_2021.filter(col("removed_by_mod") == True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text preprocessing, making the body column ready for TF-IDF and sentiment model

# COMMAND ----------

from sparknlp.base import Finisher
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.functions import avg, col

documentAssembler = DocumentAssembler() \
.setInputCol("body") \
.setOutputCol("document")

sentenceDetector = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

stop_words = StopWordsCleaner() \
.setInputCols(["token"]) \
.setOutputCol("cleanTokens")

pipeline = PretrainedPipeline("check_spelling", "en")

lemmatizer = LemmatizerModel.pretrained() \
.setInputCols(["cleanTokens"]) \
.setOutputCol("lemma")

normalizer = Normalizer() \
.setInputCols(["lemma"]) \
.setOutputCol("normal") \
.setLowercase(True) \
.setCleanupPatterns([("""[^A-Za-z]""")])

tokenAssembler = TokenAssembler() \
.setInputCols("document", "normal") \
.setOutputCol("cleanText")

embedding = WordEmbeddingsModel.pretrained() \
.setInputCols(["document","normal"]) \
.setOutputCol("embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
.setInputCols("embeddings") \
.setOutputCols("finished_sentence_embeddings") \
.setOutputAsVector(True) \
.setCleanAnnotations(False)  \

finisher = Finisher() \
     .setInputCols(['normal',"cleanText"])

pipeline = Pipeline() \
.setStages([
documentAssembler,
sentenceDetector,
tokenizer,
stop_words,
lemmatizer,
normalizer,
tokenAssembler,
embedding,
embeddingsFinisher,
finisher
])

result_token = pipeline.fit(sampled_not_removed).transform(sampled_not_removed)
result_token.show()

# COMMAND ----------

sampled_just_deleted = sampled_not_removed.filter((col("author") == "[deleted]") & (col("body") != "[removed]"))

pipeline_2 = Pipeline() \
.setStages([
documentAssembler,
sentenceDetector,
tokenizer,
stop_words,
lemmatizer,
normalizer,
tokenAssembler,
embedding,
embeddingsFinisher,
finisher
])

result_token_deleted = pipeline_2.fit(sampled_just_deleted).transform(sampled_just_deleted)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TF-IDF

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF
tfizer = CountVectorizer(inputCol='finished_normal',
                         outputCol='tf_features')
tf_model = tfizer.fit(result_token_deleted)
tf_result = tf_model.transform(result_token_deleted)
idfizer = IDF(inputCol='tf_features', 
              outputCol='tf_idf_features')
idf_model = idfizer.fit(tf_result)
tfidf_result_2021 = idf_model.transform(tf_result)

# COMMAND ----------

vocabList = tf_model.vocabulary

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

tfidf_2021_c = tfidf_result_2021.select('tf_idf_features')
# create two new colnum
tfidf_2021_c = tfidf_2021_c.withColumn("array",vector_to_array("tf_idf_features"))\
.withColumn("max_index_col",F.expr("array_position(array,array_max(array))"))
# count frequency
key_word_index_2021 = tfidf_2021_c.groupBy('max_index_col').count().orderBy(col("count"), ascending=False)

# COMMAND ----------

from pyspark.sql.functions import udf
def vocab(i):
    return vocabList[i]
udf_word = udf(lambda x:vocab(x))
key_word_index_2021 = key_word_index_2021.withColumn("word",udf_word(col("max_index_col")))
key_word_index_2021.show(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manually extract 10 keywords from the top 30 common words for feature engeneering

# COMMAND ----------

common_deleted_words = ['nothing', 'fuck','interest','wrong','follow','hate','never','face','issue','dick']


# COMMAND ----------

st_df = result_token.withColumn("cleanText", result_token["finished_cleanText"].getItem(0))
st_df.select("finished_normal").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate new boolean feature of body containing any of the 10 keywords above 

# COMMAND ----------

from pyspark.sql.functions import arrays_overlap, array
import pyspark.sql.functions as F

st_df = st_df.withColumn("common_deleted_words", F.array([F.lit(x) for x in common_deleted_words]))
st_df = st_df.withColumn("contain_often_deleted_words", arrays_overlap(st_df.finished_normal, st_df.common_deleted_words))
st_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy sentiment model to analyze the sentiment of body texts

# COMMAND ----------

import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

# Define Spark NLP pipleline
documentAssembler = DocumentAssembler()\
    .setInputCol("cleanText")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")


sentimentdl = SentimentDLModel.pretrained(name='sentimentdl_use_twitter', lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])

# COMMAND ----------

# Run the pipeline
pipelineModel = nlpPipeline.fit(st_df)
st_result = pipelineModel.transform(st_df)
# drop redundant columns
st_result = st_result.drop('finished_cleanText','document','sentence_embeddings','common_deleted_words')
# Pull out the sentiment output into its own column in the main dataframe
st_result=st_result.withColumn('sentiment_class', F.concat_ws('',col('sentiment.result')))
st_result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mark removed posts with unknown sentiment and keyword, then assemble them back to form data for ML 

# COMMAND ----------

st_result_ml = st_result.withColumn("deleted", (col("author") == "[deleted]") & (col("body") != "[deleted]"))
st_result_ml = st_result_ml.drop('author','body','created_utc','finished_sentence_embeddings','finished_normal','cleanText','sentiment')
st_result_ml.printSchema()

# COMMAND ----------

sampled_removed_final = sampled_removed.withColumn("contain_often_deleted_words", lit("unknown"))
sampled_removed_final = sampled_removed_final.withColumn("sentiment_class", lit("unknown"))
sampled_removed_final = sampled_removed_final.withColumn("deleted", (col("author") == "[deleted]") & (col("body") != "[deleted]"))
sampled_removed_final = sampled_removed_final.drop('author','body','created_utc')
sampled_removed_final.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preprocessing column types to make them ready for ML pipeline

# COMMAND ----------

from pyspark.sql.types import StringType,BooleanType,DateType

st_result_final = st_result_ml.withColumn("contain_often_deleted_words",col("contain_often_deleted_words").cast(StringType()))
ml_df = st_result_final.union(sampled_removed_final)
ml_df.show()

# COMMAND ----------

ml_df.groupBy("edited").count().show()

# COMMAND ----------

ml_df = ml_df.withColumn("edited", when(ml_df.edited == "false" ,False).otherwise(True))

# COMMAND ----------

ml_df.printSchema()

# COMMAND ----------

from pyspark.sql.types import *

ml_df = ml_df.withColumn("controversiality", col("controversiality").cast(IntegerType())).withColumn("gilded", col("gilded").cast(IntegerType()))
ml_df.printSchema()

# COMMAND ----------

ml_df.write.mode('Overwrite').parquet('dbfs:/FileStore/askreddit_delete_data.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Machine Learning Stage

# COMMAND ----------

from pyspark.sql.functions import *

ml_data = spark.read.parquet('dbfs:/FileStore/askreddit_delete_data.parquet')
ml_data = ml_data.withColumn("deleted", col("deleted").cast(StringType()))
ml_data = ml_data.filter(col("sentiment_class") != "")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the dataset into train, test and prediction set

# COMMAND ----------

train_data, test_data, predict_data = ml_data.randomSplit([0.8, 0.18, 0.02], 685)
print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))
print("Number of prediction records : " + str(predict_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup the ML pipeline
# MAGIC #### data->indexer->encoder->vectorAssembler->Model->labelConverter->result dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Random forest

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline, Model

# COMMAND ----------

stringIndexer_deleted = StringIndexer(inputCol="deleted", outputCol="delete_status")
stringIndexer_common = StringIndexer(inputCol="contain_often_deleted_words", outputCol="sensitive_words")
stringIndexer_sentiment = StringIndexer(inputCol="sentiment_class", outputCol="sentiment")

# COMMAND ----------

onehot_common = OneHotEncoder(inputCol="sensitive_words", outputCol="sensitive_words_vec")
onehot_sentiment = OneHotEncoder(inputCol="sentiment", outputCol="sentiment_vec")
vectorAssembler_features = VectorAssembler(
    inputCols=["controversiality", "edited","gilded","is_submitter","score","stickied","removed_by_mod","sensitive_words_vec","sentiment_vec"], 
    outputCol= "features")

# COMMAND ----------

rf = RandomForestClassifier(labelCol="delete_status", featuresCol="features", numTrees=50)

# COMMAND ----------

labelConverter = IndexToString(inputCol="prediction", 
                               outputCol="predictedDeletedStatus", 
                               labels= ["Account_deleted","Account_alive"])

# COMMAND ----------

pipeline_rf = Pipeline(stages=[stringIndexer_deleted, 
                               stringIndexer_common, 
                               stringIndexer_sentiment,  
                               onehot_common,
                               onehot_sentiment,
                               vectorAssembler_features, 
                               rf, labelConverter])

# COMMAND ----------

model_rf = pipeline_rf.fit(train_data)
pred_train = model_rf.transform(train_data)
pred_train.take(10)

# COMMAND ----------

predictions = model_rf.transform(test_data)
evaluatorRF = MulticlassClassificationEvaluator(labelCol="delete_status", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)
print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC That's already a good prediction accuracy. But can we learn further from feature importances?

# COMMAND ----------

model_rf.stages[-2].featureImportances

# COMMAND ----------

import pandas as pd
import os

PLOT_DIR = os.path.join("..","..","data", "plots", "ml_analysis_2")
CSV_DIR = os.path.join("..","..","data", "csv", "ml_analysis_2")

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
importance_rf = ExtractFeatureImp(model_rf.stages[-2].featureImportances, model_rf.transform(train_data), "features")

# COMMAND ----------

importance_rf

# COMMAND ----------

# MAGIC %md
# MAGIC Here we can see the removed-or-not features are the most important for user-cancellation. Maybe the users getting cancelled usually post inappropriate contents and cancellation is one of the final punishment by the Reddit moderators.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Linear support vector machine

# COMMAND ----------

svm = LinearSVC(labelCol="delete_status", featuresCol="features")
pipeline_svm = Pipeline(stages=[stringIndexer_deleted, 
                               stringIndexer_common, 
                               stringIndexer_sentiment,  
                               onehot_common,
                               onehot_sentiment,
                               vectorAssembler_features, 
                               svm, labelConverter])
model_svm = pipeline_svm.fit(train_data)
predictions_svm = model_svm.transform(test_data)
evaluatorRF = MulticlassClassificationEvaluator(labelCol="delete_status", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions_svm)
print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC Similar accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Gradient boosting trees

# COMMAND ----------

gbt = GBTClassifier(labelCol="delete_status", featuresCol="features", maxIter=10)
pipeline_gbt = Pipeline(stages=[stringIndexer_deleted, 
                               stringIndexer_common, 
                               stringIndexer_sentiment,  
                               onehot_common,
                               onehot_sentiment,
                               vectorAssembler_features, 
                               gbt, labelConverter])
model_gbt = pipeline_gbt.fit(train_data)
predictions_gbt = model_gbt.transform(test_data)
evaluator_gbt = MulticlassClassificationEvaluator(labelCol="delete_status", predictionCol="prediction", metricName="accuracy")
accuracy_gbt = evaluator_gbt.evaluate(predictions_gbt)
print("Accuracy = %g" % accuracy_gbt)
print("Test Error = %g" % (1.0 - accuracy_gbt))

# COMMAND ----------

# MAGIC %md
# MAGIC This one has a bit higher accuracy than the previous 2, but other metrics still need to be evaluated for further comparison.

# COMMAND ----------

evaluator_ac = MulticlassClassificationEvaluator(labelCol="delete_status", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="delete_status", predictionCol="prediction", metricName="f1")
evaluator_roc = BinaryClassificationEvaluator(labelCol="delete_status", rawPredictionCol="prediction", metricName="areaUnderROC")

# COMMAND ----------

perf_ac = []
perf_f1 = []
perf_roc = []
for model in [model_rf, model_svm, model_gbt]:
    predictions = model.transform(test_data)
    perf_ac.append(evaluator_ac.evaluate(predictions))
    perf_f1.append(evaluator_f1.evaluate(predictions))
    perf_roc.append(evaluator_roc.evaluate(predictions))
model_performances = pd.DataFrame(list(zip(perf_ac,perf_f1,perf_roc)), index = ["rf","svm","gbt"], columns=["accuracy","f1_score","ROC_area"])
model_performances

# COMMAND ----------

# MAGIC %md
# MAGIC This table contains the performances of 3 different machine learning model on our test dataset, measured in 3 different metrics(accuracy, f1 score and area under ROC curve).<br>
# MAGIC Here we could observe that:<br>
# MAGIC 1. As for accuracy, the Gradient Boosting Tree model receives the highest accuracy, a little over the other 2.<br>
# MAGIC 2. As for f1-score, the GBT model still has the highest statistics, possibly by its relatively higher precision.<br>
# MAGIC 3. For area under ROC curve, the GBT model comes to its shortcoming, and the first model of Random Forest has the largest area under its ROC curve.<br>
# MAGIC 4. To conclude from these statistics, Random Forest model seems to have a better capability at separating the classes, while the GBT model has the highest actual precision statistics. There is no significant performance difference between the 3 models, and more information may need to be extracted out by analyzing their feature importances.

# COMMAND ----------

fpath = os.path.join(CSV_DIR, "performances.csv")
model_performances.to_csv(fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Feature importances

# COMMAND ----------

print("Coefficients: " + str(model_svm.stages[-2].coefficients))
print("Intercept: " + str(model_svm.stages[-2].intercept))

# COMMAND ----------

ExtractFeatureImp(model_rf.stages[-2].featureImportances, model_rf.transform(train_data), "features")

# COMMAND ----------

import numpy as np

features_actual = ["controversiality","edited","gilded","is_submitter","score","stickied","removed_by_mod","sensitive_words_vec_false","sensitive_words_vec_unknown","sentiment_vec_positive","sentiment_vec_negative","sentiment_vec_unknown"]
svm_df = pd.DataFrame(list(zip(features_actual, np.abs(model_svm.stages[-2].coefficients))),columns=["name","score"])
svm_df = svm_df.sort_values(by=["score"], ascending=False)
svm_df

# COMMAND ----------


model_gbt.stages[-2].featureImportances.toArray()

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

feature_importances = pd.DataFrame(list(zip(features_actual, model_rf.stages[-2].featureImportances, np.abs(model_svm.stages[-2].coefficients), model_gbt.stages[-2].featureImportances)),
                                  columns=["feature","rf_imp","svm_imp","gbt_imp"])
fpath = os.path.join(CSV_DIR, "importances.csv")
feature_importances.to_csv(fpath)
feature_importances

# COMMAND ----------

# MAGIC %md
# MAGIC Here we could observe that while the importances of 3 models disagree on the other features, they all recognize the importance of removed-related features, indicating that account cancellation often happens with removed-by-mod posts, possibly because the moderators forcibly cancelled those accounts due to their posting inappropriate contents.

# COMMAND ----------

scaled_importances = pd.DataFrame(MinMaxScaler().fit(feature_importances[["rf_imp","svm_imp","gbt_imp"]]).transform(feature_importances[["rf_imp","svm_imp","gbt_imp"]]),columns=["rf_imp","svm_imp","gbt_imp"])

# COMMAND ----------

import matplotlib.pyplot as plt 

plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(features_actual))*1.5
rf_imp = list(scaled_importances["rf_imp"])
rf_imp.reverse()
svm_imp = list(scaled_importances["svm_imp"])
svm_imp.reverse()
gbt_imp = list(scaled_importances["gbt_imp"])
gbt_imp.reverse()
ax.barh(y_pos - 0.4, rf_imp, 0.4, align='center', label = 'Random Forest')
ax.barh(y_pos, svm_imp, 0.4, align='center', label = 'Linear SVC')
ax.barh(y_pos + 0.4, gbt_imp, 0.4, align='center', label = 'Gradient Boosting Trees')
ax.set_yticks(y_pos)
y_labels = features_actual
y_labels.reverse()
ax.set_yticklabels(y_labels)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Feature importances by 3 models')
ax.legend()
fpath = os.path.join(PLOT_DIR, "feature_importances.png")
plt.savefig(fpath)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This bar chart describes the relative importances of features utlized in the 3 machine learning models. The importances of Linear SVC model are the absolute values(there were negative contributions), and the importance values have been normalized into a range of [0,1]; we could observe the following findings:<br>
# MAGIC 1. Regarding the overall distribution of the importances, we could see that Linear SVC has the most balanced scale of importances, followed by Random Forest; as for the Gradient Boosting Trees model of green bars, the importances seem to concentrate on a few features.<br>
# MAGIC 2. Regarding the single features, we shall see that the unknown features(sentiment_vec_known, sensitive_words_vec_unknown) and removed_by_mod are generally having the largest importances in prediction. Since posts are tagged unknown in sentiment and sensitive words if and only if their body contents got removed by the moderators, these features all indicate that regulation of moderators could be a major factor in the cancellation of Reddit accounts, which seems reasonable since the sensitive words are usually offensive and moderators seem to have the power to directly cancel accounts posting too much inappropriate contents. <br>
# MAGIC 3. As the main ways to measure the value of a post, score and gilded seem to have little importance in predicting account cancellation. This might comes from a fact that contents from a cancelled user usually receive low score(some even get negative scores!) and do not worth to be gilded at all.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Save and load the GBT model

# COMMAND ----------

import os

PLOT_DIR = os.path.join("..","..","data", "plots", "ml_analysis_2")
CSV_DIR = os.path.join("..","..","data", "csv", "ml_analysis_2")
fpath = os.path.join("..","ml_analysis_2","performances.csv")
print(CSV_DIR)

# COMMAND ----------

print(os.getcwd())

# COMMAND ----------

fpath = os.path.join("/Repos/Project/fall-2022-reddit-big-data-project-project-group-21/code/ml_analysis_2","gbt_pipeline")
model_gbt.write().overwrite().save(fpath)

# COMMAND ----------

import os
from pyspark.ml import PipelineModel

fpath = os.path.join("/Repos/Project/fall-2022-reddit-big-data-project-project-group-21/code/ml_analysis_2","gbt_pipeline")
gbt_model = PipelineModel.load(fpath)
prediction_from_file = gbt_model.transform(predict_data)
prediction_from_file.show()
