# Databricks notebook source
# MAGIC %md
# MAGIC ## NLP

# COMMAND ----------

# read comments data in wallstreetbets subreddit
comments = spark.read.parquet("dbfs:/FileStore/comments_wallstreetbets.parquet") 

# COMMAND ----------

# set saving address
import os
PLOT_DIR = os.path.join("..","data", "plots")
CSV_DIR = os.path.join("..","data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

# check the shape of the data
comments_row_count = comments.count()
comment_col_count = len(comments.columns)
print(f"shape of the comments dataframe is {comments_row_count:,}x{comment_col_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter data

# COMMAND ----------

#filter the removed and deleted body
import pyspark.sql.functions as F
comments = comments.where((F.col("body") != '[removed]')&(F.col("body") != '[deleted]'))
# filter delete the row with '![img]' and "I am a bot from /r/wallstreetbets. Your submission was removed because was too short.""
comments = comments.withColumn('body', F.regexp_replace('body',r'(\!\[img\]\(emote\|t5_2th52\|\d+\))',''))\
                         .where(F.col("body") != '')
del_string="""I am a bot from /r/wallstreetbets. Your submission was removed because was too short. Please make it a comment if it's not worth expounding on."""
comments = comments.where(F.col("body") != del_string)

# COMMAND ----------

# check the shape of the data
comments_row_count = comments.count()
comment_col_count = len(comments.columns)
print(f"shape of the comments dataframe is {comments_row_count:,}x{comment_col_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Merge exteral data

# COMMAND ----------

# import external data
# import stock market index data
# external dataset contains each day index of S&P 500, Nasdap and Dow Jones from 2021-01-01 to 2022-08-31
# the missing value of stock market closed day is filled by valid data the day before
import pandas as pd
stock_index = pd.read_csv('/Workspace/Repos/Project/fall-2022-reddit-big-data-project-project-group-21/data/stock_index.csv')
# clean
stock_index = stock_index.drop('Unnamed: 0',axis=1)
stock_index.rename(columns={"date":"d"},inplace =True)
stock_index

# COMMAND ----------

# convert pandas dataframe to pyspark dataframe
stock = spark.createDataFrame(stock_index) 

# COMMAND ----------

# merge external data with comment data on date

from pyspark.sql.functions import *
# convert date column in stock to date format
stock = stock.withColumn('d', to_date(to_timestamp('d', 'yyyy-MM-dd').cast('timestamp'),"yyyy-MM-dd HH:mm:ss"))
# merge on dat
comment_new = comments.join(stock, comments.date == stock.d, 'left').drop('d').cache()

# COMMAND ----------

# check the shape of the data
comments_row_count = comment_new.count()
comment_col_count = len(comment_new.columns)
print(f"shape of the comments dataframe is {comments_row_count:,}x{comment_col_count}")

# COMMAND ----------

# group data by date
import pandas as pd
from pyspark.sql.functions import *
comment_by_date = comment_new.groupBy("date").agg(count("body").alias("comment_count"), \
                                                  avg("sp500").alias("sp500"), \
                                                  avg("nasdaq").alias("nasdaq"), 
                                                  avg("dj").alias("dj")).orderBy(col("date"), ascending=True).collect()
comment_by_date = spark.createDataFrame(comment_by_date).toPandas()

# COMMAND ----------

### for this plot, you can drag and select a region to zoom in 

# COMMAND ----------

import plotly.graph_objects as go
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

count = comment_by_date['comment_count']
date = comment_by_date['date']

fig = make_subplots(rows=3, cols=2,
                    subplot_titles=("<b>from 2021-01 to 2022-08</b>", "<b>from 2021-04 to 2022-08</b>", "", "", "", ""),
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    horizontal_spacing=0.16,
                    specs=[[{"secondary_y": True},{"secondary_y": True}], \
                           [{"secondary_y": True},{"secondary_y": True}],[{"secondary_y": True},{"secondary_y": True}]])

# sp500
fig.add_trace(
    go.Scatter(x=date, y=count, name="daily comment number",line=dict(color="#f38818")),
    row=1, col=1, secondary_y=False)

fig.add_trace(
    go.Scatter(x=date, y=comment_by_date['sp500'], name="sp500 index",line=dict(color="#7d90ff")),
    row=1, col=1, secondary_y=True
)

# Tnasdaq
fig.add_trace(
    go.Scatter(x=date, y=count,name="daily comment number", line=dict(color="#f38818"),showlegend=False),
    row=2, col=1, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date, y=comment_by_date['nasdaq'], name="Nasdaq index",line=dict(color="#f599ab")),
    row=2, col=1, secondary_y=True,
)

# dj
fig.add_trace(
    go.Scatter(x=date, y=count, name="daily comment number",line=dict(color="#f38818"),showlegend=False),
    row=3, col=1, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date, y=comment_by_date['dj'], name="Dow Jones index",line=dict(color="#e6d59c")),
    row=3, col=1, secondary_y=True,
)

# -------------------------------------------------------------------------------------------------
# sp500
fig.add_trace(
    go.Scatter(x=date[100:], y=count[100:], name="daily comment number",line=dict(color="#f38818"),showlegend=False),
    row=1, col=2, secondary_y=False)

fig.add_trace(
    go.Scatter(x=date[100:], y=comment_by_date['sp500'][100:], name="sp500 index",line=dict(color="#7d90ff"),showlegend=False),
    row=1, col=2, secondary_y=True,
)

# Tnasdaq
fig.add_trace(
    go.Scatter(x=date[100:], y=count[100:],name="daily comment number", line=dict(color="#f38818"),showlegend=False),
    row=2, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date[100:], y=comment_by_date['nasdaq'][100:], name="Nasdaq index",line=dict(color="#f599ab"),showlegend=False),
    row=2, col=2, secondary_y=True,
)

# dj
fig.add_trace(
    go.Scatter(x=date[100:], y=count[100:], name="daily comment number",line=dict(color="#f38818"),showlegend=False),
    row=3, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date[100:], y=comment_by_date['dj'][100:], name="Dow Jones index",line=dict(color="#e6d59c"),showlegend=False),
    row=3, col=2, secondary_y=True,
)



fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1),title="<b>Wallstreetbets Daily Comment Count VS Stock Market Index</b>",
        title_x = 0.5,
        title_y = 0.95,
        height=600,
        margin={"t": 150, "b": 0, "l": 0, "r": 0},
        hovermode=False,
        font=dict(
        family="Droid Sans",
        size=15,
        color='#696969'))
fig.update_xaxes(title_text='<b>Date</b>', title_font=dict(size=12),row=3)
fig.update_yaxes(title_text='<b>Daily Comment Count</b>', title_font=dict(size=12), row=2,col=1)
fig.update_yaxes(title_text='<b>Stock Market Index</b>', title_font=dict(size=12), row=2,col=2)



## Save the plot in the plot dir so that it can be checked in into the repo
fig.show()
fpath = os.path.join(PLOT_DIR, "comment_stockindex.html")
fig.write_html(fpath)

# COMMAND ----------

# save to DBFS
comment_new.write.mode('Overwrite').parquet('dbfs:/FileStore/comments_merged.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regex

# COMMAND ----------

# MAGIC %md
# MAGIC ### Character matches

# COMMAND ----------

comment_new = spark.read.parquet("dbfs:/FileStore/comments_merged.parquet") 

# COMMAND ----------

comments_new_column = comment_new.withColumn("dollar_sign", comment_new["body"].rlike("\$"))
comments_new_column = comments_new_column.withColumn("ellipsis", comments_new_column["body"].rlike("(\.{3})"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Keyword matches

# COMMAND ----------

labor_words = ["wage", "salary", "employ", "unemploy", "job", "labor", "labour"]
regex_labor = "(?i)\\b"+"|\\b".join(labor_words)
comments_new_column = comments_new_column.withColumn("labor", comments_new_column["body"].rlike(regex_labor))

# COMMAND ----------

yolo_words = ["yolo", "all in", "life saving", "leverage"]
regex_yolo = "(?i)\\b"+"|\\b".join(yolo_words)
comments_new_column = comments_new_column.withColumn("yolo", comments_new_column["body"].rlike(regex_yolo))

# COMMAND ----------

btd_words = ["btd", "btfd"]
regex_btd = "(?i)\\b"+"|\\b".join(btd_words)
comments_new_column = comments_new_column.withColumn("btd", comments_new_column["body"].rlike(regex_btd))

# COMMAND ----------

from pyspark.sql.functions import *
comments_new_column_year = comments_new_column.withColumn("year", year(from_unixtime(col('created_utc'))))
keyword_sum = comments_new_column_year.groupby("year").agg(avg(col("dollar_sign").cast("int")).alias("dollar"),
                                  avg(col("ellipsis").cast("int")).alias("ellipsis"),
                                  avg(col("labor").cast("int")).alias("labor"),
                                  avg(col("yolo").cast("int")).alias("yolo"),
                                  F.avg(col("btd").cast("int")).alias("btd")).toPandas()
keyword_sum.to_csv("../data/csv/keyword_sum.csv")
keyword_sum

# COMMAND ----------

# filter data with dollar sign
df_dollar = comments_new_column.filter(col("dollar_sign"))
# group data by date
df_dollar_by_date = df_dollar.filter(col("dollar_sign")).groupBy("date").agg(count("body").alias("comment_count"), \
                                                  avg("sp500").alias("sp500"), \
                                                  avg("nasdaq").alias("nasdaq"), 
                                                  avg("dj").alias("dj")).orderBy(col("date"), ascending=True).collect()
df_dollar_by_date = spark.createDataFrame(df_dollar_by_date).toPandas()

# COMMAND ----------

### for this plot, you can drag and select a region to zoom in 

# COMMAND ----------

count = df_dollar_by_date['comment_count']
date = df_dollar_by_date['date']

fig = make_subplots(rows=3, cols=2,
                    subplot_titles=("<b>from 2021-01 to 2022-08</b>", "<b>from 2021-04 to 2022-08</b>", "", "", "", ""),
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    horizontal_spacing=0.16,
                    specs=[[{"secondary_y": True},{"secondary_y": True}], \
                           [{"secondary_y": True},{"secondary_y": True}],[{"secondary_y": True},{"secondary_y": True}]])

# sp500
fig.add_trace(
    go.Scatter(x=date, y=count, name="daily comment number",line=dict(color="#f38818")),
    row=1, col=1, secondary_y=False)

fig.add_trace(
    go.Scatter(x=date, y=df_dollar_by_date['sp500'], name="sp500 index",line=dict(color="#7d90ff")),
    row=1, col=1, secondary_y=True
)

# Tnasdaq
fig.add_trace(
    go.Scatter(x=date, y=count,name="daily comment number", line=dict(color="#f38818"),showlegend=False),
    row=2, col=1, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date, y=df_dollar_by_date['nasdaq'], name="Nasdaq index",line=dict(color="#f599ab")),
    row=2, col=1, secondary_y=True,
)

# dj
fig.add_trace(
    go.Scatter(x=date, y=count, name="daily comment number",line=dict(color="#f38818"),showlegend=False),
    row=3, col=1, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date, y=df_dollar_by_date['dj'], name="Dow Jones index",line=dict(color="#e6d59c")),
    row=3, col=1, secondary_y=True,
)

# -------------------------------------------------------------------------------------------------
# sp500
fig.add_trace(
    go.Scatter(x=date[100:], y=count[100:], name="daily comment number",line=dict(color="#f38818"),showlegend=False),
    row=1, col=2, secondary_y=False)

fig.add_trace(
    go.Scatter(x=date[100:], y=df_dollar_by_date['sp500'][100:], name="sp500 index",line=dict(color="#7d90ff"),showlegend=False),
    row=1, col=2, secondary_y=True,
)

# Tnasdaq
fig.add_trace(
    go.Scatter(x=date[100:], y=count[100:],name="daily comment number", line=dict(color="#f38818"),showlegend=False),
    row=2, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date[100:], y=df_dollar_by_date['nasdaq'][100:], name="Nasdaq index",line=dict(color="#f599ab"),showlegend=False),
    row=2, col=2, secondary_y=True,
)

# dj
fig.add_trace(
    go.Scatter(x=date[100:], y=count[100:], name="daily comment number",line=dict(color="#f38818"),showlegend=False),
    row=3, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=date[100:], y=df_dollar_by_date['dj'][100:], name="Dow Jones index",line=dict(color="#e6d59c"),showlegend=False),
    row=3, col=2, secondary_y=True,
)



fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1),title="<b>Wallstreetbets Daily Comment Count (with Dollar Sign) VS Stock Market Index</b>",
        title_x = 0.5,
        title_y = 0.95,
        height=600,
        margin={"t": 150, "b": 0, "l": 0, "r": 0},
        hovermode=False,
        font=dict(
        family="Droid Sans",
        size=15,
        color='#696969'))
fig.update_xaxes(title_text='<b>Date</b>', title_font=dict(size=12),row=3)
fig.update_yaxes(title_text='<b>Daily Comment Count</b>', title_font=dict(size=12), row=2,col=1)
fig.update_yaxes(title_text='<b>Stock Market Index</b>', title_font=dict(size=12), row=2,col=2)



## Save the plot in the plot dir so that it can be checked in into the repo
fig.show()
fpath = os.path.join(PLOT_DIR, "comment_stockindex(dollar).html")
fig.write_html(fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reduce data size by filters

# COMMAND ----------

from pyspark.sql.functions import *
comments_new_column_year = comments_new_column.withColumn("year", year(from_unixtime(col('created_utc'))))
data_distribution = comments_new_column_year.groupby("year").agg(avg(col("dollar_sign").cast("int")).alias("dollar"),
                                  avg(col("ellipsis").cast("int")).alias("ellipsis"),
                                  avg(col("labor").cast("int")).alias("labor"),
                                  avg(col("yolo").cast("int")).alias("yolo"),
                                  F.avg(col("btd").cast("int")).alias("btd")).toPandas()
data_distribution

# COMMAND ----------

# MAGIC %md
# MAGIC This summary table displays the ratio of several stock-related topics among the WallStreetBets Subreddit comments in 2021 and 2022.<br>
# MAGIC First of all, the ratios of those topics are not as big as a crucial part of the game, possibly indicating that comments tend to be more related to personal communication rather than direct references on stock market actions where people usually use those topics to describe their actions.<br>
# MAGIC Secondly, we could see that ratios of comments including dollar sign($), ellipsis(...), yolo(you can only live once) and btd(buy the dip) all dropped in 2022, from ~30% to ~50%. If we regard dollar sign as stock-related, ellipsis as people's awkward/sad attitude, yolo and btd as indicator of stock market behavior, then the dropping usage of those keywords might lead to a hypothesis that Reddit users in this Subreddit are having a lower interest on investing in the stock market, while a raise in labor-related topic's popularity might indicate that people are concerning more on the labor market rather than stock market. This hypothesis is tempting since it matches the fact that world economy gets further worsened by the long-lived pandemic of COVID-19 in 2022, when people possibly find themselves more needy for a steady job rather than investing their budget into the volatile stock market.

# COMMAND ----------

comments_new_column.createOrReplaceTempView('comment_vw')
results1 = spark.sql('select avg(score) as mean_score, percentile(score, 0.01) as Q1, percentile(score, 0.25) as Q25, \
                     percentile(score, 0.5) as median, percentile(score, 0.75) as Q75, percentile(score, 0.99) as Q99 \
                     from comment_vw')
results1.show()

# COMMAND ----------

# high score comment 95%-99% with dollar sign
high_score_boundary = spark.sql('select percentile(score, 0.95) as Q95, percentile(score, 0.99) as Q99 \
                     from comment_vw').toPandas()
high_score_boundary

# COMMAND ----------

comment_highscore = comments_new_column.filter(col("dollar_sign")).filter((col('score') >= 20) & (col('score') <= 83))
comment_highscore.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Text clean

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

result_token = pipeline.fit(comment_highscore).transform(comment_highscore)
result_token.show()

# COMMAND ----------

# save to DBFS
result_token.write.mode('Overwrite').parquet('dbfs:/FileStore/text_clean.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conduct basic data text checks/analysis

# COMMAND ----------

result_token.count()

# COMMAND ----------

result_token = spark.read.parquet("dbfs:/FileStore/text_clean.parquet") 
import pyspark.sql.functions as F

# COMMAND ----------

import pyspark.sql.functions as F
# calculate the lengh of body and create a new column to stored it
result_token1 = result_token.withColumn("length_of_body", F.size("finished_normal")) \
.withColumn("year", year(from_unixtime(col('created_utc'))))

# COMMAND ----------

#summary table for body lenghs
result_token1.createOrReplaceTempView('result_token_vw')
result_token_body_lenghs_summary = spark.sql('select year, MIN(length_of_body) as min,percentile(length_of_body, 0.01) as Q1, percentile(length_of_body, 0.25) as Q25, \
                     percentile(length_of_body, 0.5) as median,round(avg(length_of_body),2) as mean, percentile(length_of_body, 0.75) as Q75, percentile(score, 0.99) as Q99, Max(score) as max \
                     from result_token_vw \
                     group by year').toPandas()
result_token_body_lenghs_summary

# COMMAND ----------

#summary table for controversiality
result_token_controversiality_summary = spark.sql('select year, avg(controversiality) as percentage \
                     from result_token_vw \
                     group by year').toPandas()
result_token_controversiality_summary

# COMMAND ----------

#summary table for score
result_token_score_summary = spark.sql('select year, MIN(score) as min,percentile(score, 0.01) as Q1, percentile(score, 0.25) as Q25, \
                     percentile(score, 0.5) as median,round(avg(score),2) as mean, percentile(score, 0.75) as Q75, percentile(score, 0.99) as Q99, Max(score) as max \
                     from result_token_vw \
                     group by year').toPandas()
result_token_score_summary

# COMMAND ----------

import pandas as pd
summary_df = pd.concat([result_token_body_lenghs_summary, result_token_score_summary,result_token_controversiality_summary],axis = 0)
summary_df['feature_name'] = ['comment words number','comment words number','comment score','comment score','controversiality','controversiality']
summary_df = summary_df[['feature_name','year', 'min', 'Q1', 'Q25', 'median', 'mean', 'Q75', 'Q99', 'max','percentage']]
summary_df

# COMMAND ----------

# save summary table
fpath = os.path.join(CSV_DIR, "features_quantile.csv")
summary_df.to_csv(fpath)

# COMMAND ----------

summary_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of text lenghs

# COMMAND ----------

# MAGIC %md
# MAGIC As we can find from this summary table, the most frequent text lenghs is 27. Let's take a look at some rows which body lengh are 27.

# COMMAND ----------

len_table = result_token1.groupBy("length_of_body")\
           .agg(F.count('length_of_body').alias('body_text_lengh_count'))\
           .sort("length_of_body").toPandas()
len_table

# COMMAND ----------

result_token1.where(F.col('length_of_body')==0).select('body','finished_normal').show(30)

# COMMAND ----------

result_token1.groupBy("length_of_body")\
           .agg(F.count('length_of_body').alias('body_text_lengh_count'))\
           .sort(F.desc("body_text_lengh_count")).show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# plot text lengh distribution 
plt.figure()
plt.rcParams['figure.figsize'] = [15, 12]
plt.subplot(211)
plt.rcParams['axes.facecolor']='#f3f6f7'
plt.grid(True,linestyle=':',color='gray')
plt.bar(range(400),len_table['body_text_lengh_count'][:400],color = "#f38818")
#plt.xlabel("Number of words in comment", fontsize = 15,fontweight='bold',labelpad=30)
plt.ylabel("Count", fontsize = 15,fontweight='bold',labelpad=30,color ='#696969')
plt.text(200,2000,'Text lengths smaller than 400 words', fontsize = 20, fontweight ='bold',color='gray')
plt.title('Distribution of Text Lengths',fontsize = 25, fontweight='bold',pad=30,color='#696969')
 
plt.subplot(212)
plt.grid(True,linestyle=':',color='gray')
plt.bar(range(100),len_table['body_text_lengh_count'][:100],color = "#f38818")
plt.xlabel("Number of words in comment", fontsize = 15,fontweight='bold',labelpad=30,color ='#696969')
plt.ylabel("Count", fontsize = 15,fontweight='bold',labelpad=30,color ='#696969')
plt.text(50,2000,'Text lengths smaller than 100 words', fontsize = 20, fontweight ='bold',color='gray')
plt.tight_layout()
 
## Save the plot in the plot dir so that it can be checked in into the repo
plot_fpath = os.path.join(PLOT_DIR, 'distribution_text_length.png')
plt.savefig(plot_fpath)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top words in comment

# COMMAND ----------

#pipeline for tfidf
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
vectorizer = CountVectorizer(inputCol='finished_normal', outputCol='tf_features')
idf = IDF(inputCol='tf_features', outputCol='tf_idf_features')
pipeline = Pipeline(stages=[vectorizer, idf])
model = pipeline.fit(result_token1)

# COMMAND ----------

#bl
result_token1_2021 = result_token1.filter(year(from_unixtime(col('created_utc'))) == 2021)
result_token1_2022 = result_token1.filter(year(from_unixtime(col('created_utc'))) == 2022)
result_token1_2022.show()


# COMMAND ----------

import numpy as np

total_counts = model.transform(result_token1_2021)\
                    .select('tf_features').rdd\
                    .map(lambda row: row['tf_features'].toArray())\
                    .reduce(lambda x,y: [x[i]+y[i] for i in range(len(y))])

vocabList = model.stages[0].vocabulary
d = {'vocabList':vocabList,'counts':total_counts}

freq_2021 = spark.createDataFrame(np.array(list(d.values())).T.tolist(),list(d.keys()))
freq_2021.show()

# COMMAND ----------

fpath = os.path.join(CSV_DIR, "topword_2021.csv")
freq_2021.toPandas().to_csv(fpath)
fpath = os.path.join(CSV_DIR, "topword_2022.csv")
freq_2022.toPandas().to_csv(fpath)

# COMMAND ----------

total_counts = model.transform(result_token1_2021)\
                    .select('tf_features').rdd\
                    .map(lambda row: row['tf_features'].toArray())\
                    .reduce(lambda x,y: [x[i]+y[i] for i in range(len(y))])
# vocabruary
vocabList = model.stages[0].vocabulary
d = {'vocabList':vocabList,'counts':total_counts}

freq_2021 = spark.createDataFrame(np.array(list(d.values())).T.tolist(),list(d.keys()))
freq.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### TF-IDF

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer
tfizer = CountVectorizer(inputCol='finished_normal',
                         outputCol='tf_features')
tf_model = tfizer.fit(result_token1_2021)
tf_result_2021 = tf_model.transform(result_token1_2021)
tf_model = tfizer.fit(result_token1_2022)
tf_result_2022 = tf_model.transform(result_token1_2022)

# COMMAND ----------

from pyspark.ml.feature import IDF
idfizer = IDF(inputCol='tf_features', 
              outputCol='tf_idf_features')
idf_model = idfizer.fit(tf_result_2021)
tfidf_result_2021 = idf_model.transform(tf_result_2021)
idf_model = idfizer.fit(tf_result_2022)
tfidf_result_2022 = idf_model.transform(tf_result_2022)

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
tfidf_2021_c = tfidf_result_2021.select('tf_idf_features')
# create two new colnum
tfidf_2021_c = tfidf_2021_c.withColumn("array",vector_to_array("tf_idf_features"))\
.withColumn("max_index_col",F.expr("array_position(array,array_max(array))"))
# count frequency
key_word_index_2021 = tfidf_2021_c.groupBy('max_index_col').count().orderBy(col("count"), ascending=False)

tfidf_2022_c = tfidf_result_2022.select('tf_idf_features')
# create two new colnum
tfidf_2022_c = tfidf_2022_c.withColumn("array",vector_to_array("tf_idf_features"))\
.withColumn("max_index_col",F.expr("array_position(array,array_max(array))"))
# count frequency
key_word_index_2022 = tfidf_2022_c.groupBy('max_index_col').count().orderBy(col("count"), ascending=False)

# COMMAND ----------

from pyspark.sql.functions import udf
def vocab(i):
    return vocabList[i]
udf_word = udf(lambda x:vocab(x))
key_word_index_2021 = key_word_index_2021.withColumn("word",udf_word(col("max_index_col")))
key_word_index_2022 = key_word_index_2022.withColumn("word",udf_word(col("max_index_col")))

# COMMAND ----------

#save summary table
fpath = os.path.join(CSV_DIR, "tfidf_2021.csv")
key_word_index_2021.toPandas().to_csv(fpath)
fpath = os.path.join(CSV_DIR, "tfidf_2022.csv")
key_word_index_2022.toPandas().to_csv(fpath)

# COMMAND ----------

key_word_index_2021.show(10)

# COMMAND ----------

key_word_index_2022.show(10)

# COMMAND ----------

key_word_index = tfidf_r.groupBy('max_index_col').count().orderBy(col("count"), ascending=False).toPandas()
# fpath = os.path.join(CSV_DIR, f"top_{top_n}_subreddits_count_by_submissions.csv")
#top_n_subreddits_sub.to_csv(fpath)                        
key_word_index

# COMMAND ----------

from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly
import plotly.express as px
import matplotlib.pyplot as plt

# COMMAND ----------

#### for the following wordcloud, you can drag and select to zoom in small words

# COMMAND ----------

# wordcloud
fpath = os.path.join(CSV_DIR, "topword_2021.csv")
df2021 =  pd.read_csv(fpath)
fpath = os.path.join(CSV_DIR, "topword_2022.csv")
df2022 =  pd.read_csv(fpath)
fpath = os.path.join(CSV_DIR, "tfidf_2021.csv")
df2021_tfidf =  pd.read_csv(fpath)
fpath = os.path.join(CSV_DIR, "tfidf_2022.csv")
df2022_tfidf =  pd.read_csv(fpath)
max_words = 300
word_dict2021 = dict(zip(df2021[:max_words]['vocabList'].tolist(), df2021[:max_words]['counts'].tolist()))
word_dict2022 = dict(zip(df2022[:max_words]['vocabList'].tolist(), df2022[:max_words]['counts'].tolist()))
word_dict2021_tfidf = dict(zip(df2021_tfidf[:max_words]['word'].tolist(), df2021_tfidf[:max_words]['count'].tolist()))
word_dict2022_tfidf = dict(zip(df2022_tfidf[:max_words]['word'].tolist(), df2022_tfidf[:max_words]['count'].tolist()))

# COMMAND ----------

########## for wordcloud part, you can select a region and zoom in to check small words
wordcloud1 = WordCloud(background_color='#f3f6f7',scale=5, collocations = False,  \
                       max_words = max_words, max_font_size = 30, min_font_size = 3,\
                      colormap = 'rainbow',random_state=1).generate_from_frequencies(word_dict2021)
fig = go.Figure()
fig= px.imshow(wordcloud1.to_array())
fig.update_layout(
        title="<b>Wallstreetbets Most Common Words in 2021</b>",
        title_x = 0.5,
        title_y = 0.95,
        height=300,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 50, "b": 0, "l": 0, "r": 0},
        hovermode=False,
        font=dict(
        family="Droid Sans",
        size=15,
        color='#696969'
    )
    )
fig.show()
fpath = os.path.join(PLOT_DIR, "topword_2021.html")
fig.write_html(fpath)

# COMMAND ----------

wordcloud2 = WordCloud(background_color='#f3f6f7',scale=5, collocations = False,  \
                       max_words = max_words, max_font_size = 30, min_font_size = 3,\
                      colormap = 'rainbow',random_state=1).generate_from_frequencies(word_dict2022)
fig = go.Figure()
fig= px.imshow(wordcloud2.to_array())
fig.update_layout(
        title="<b>Wallstreetbets Most Common Words in 2022</b>",
        title_x = 0.5,
        title_y = 0.95,
        height=300,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 50, "b": 0, "l": 0, "r": 0},
        hovermode=False,
        font=dict(
        family="Droid Sans",
        size=15,
        color='#696969'
    )
    )
fig.show()
fpath = os.path.join(PLOT_DIR, "topword_2022.html")
fig.write_html(fpath)

# COMMAND ----------

wordcloud3 = WordCloud(background_color='#f3f6f7',scale=5, collocations = False,  \
                       max_words = max_words, max_font_size = 30, min_font_size = 3,\
                      colormap = 'rainbow',random_state=1).generate_from_frequencies(word_dict2021_tfidf)
fig = go.Figure()
fig= px.imshow(wordcloud3.to_array())
fig.update_layout(
        title="<b>Wallstreetbets Important Words in 2021</b>",
        title_x = 0.5,
        title_y = 0.95,
        height=300,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 50, "b": 0, "l": 0, "r": 0},
        hovermode=False,
        font=dict(
        family="Droid Sans",
        size=15,
        color='#696969'
    )
    )
fig.show()

fpath = os.path.join(PLOT_DIR, "topword_2021_tfidf.html")
fig.write_html(fpath)

# COMMAND ----------

wordcloud4 = WordCloud(background_color='#f3f6f7',scale=10, collocations = False,  \
                       max_words = max_words, max_font_size = 30, min_font_size = 3,\
                      colormap = 'rainbow',random_state=1).generate_from_frequencies(word_dict2022_tfidf)
fig = go.Figure()
fig= px.imshow(wordcloud4.to_array())
fig.update_layout(
        title="<b>Wallstreetbets Important Words in 2022</b>",
        title_x = 0.5,
        title_y = 0.95,
        height=300,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 50, "b": 0, "l": 0, "r": 0},
        hovermode=False,
        font=dict(
        family="Droid Sans",
        size=15,
        color='#696969'
    )
    )
fig.show()

fpath = os.path.join(PLOT_DIR, "topword_2022_tfidf.html")
fig.write_html(fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentiment Model

# COMMAND ----------

import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

st_df = result_token1.withColumn("cleanText", result_token1["finished_cleanText"].getItem(0))

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

# COMMAND ----------

# drop redundant columns
st_result = st_result.drop('finished_cleanText','document','sentence_embeddings')

# COMMAND ----------

# Pull out the sentiment output into its own column in the main dataframe
st_result=st_result.withColumn('sentiment_class', F.concat_ws('',col('sentiment.result')))

# COMMAND ----------

st_result = st_result.withColumn("year", year(from_unixtime(col('created_utc'))))

# COMMAND ----------

sentiment_sum = st_result.groupBy('year', 'sentiment_class').agg(avg(col("length_of_body").cast("int")).alias("average_length_of_body"),
                                                                 avg(col("score").cast("int")).alias("average_score"),
                                  avg(col("ellipsis").cast("int")).alias("ellipsis_percentage"),
                                  avg(col("labor").cast("int")).alias("labor_percentage"),
                                  avg(col("yolo").cast("int")).alias("yolo_percentage")).orderBy('year', 'sentiment_class').toPandas()

# COMMAND ----------

# sentiment summary table for high score comments
sentiment_sum

# COMMAND ----------

# save summary table
fpath = os.path.join(CSV_DIR, "sentiment_highs_core.csv")
sentiment_sum.to_csv(fpath)

# COMMAND ----------

st_eda_df = st_result.filter(col('sentiment_class') !='').groupBy('date','sentiment_class').agg(count('body').alias("comment_count"), \
                                                                                                avg("length_of_body").alias("avg_len"), \
                                                                                                avg("score").alias("avg_score"), \
                                                                                                avg("sp500").alias("sp500"), \
                                                                                                avg("nasdaq").alias("nasdaq"), 
                                                                                                avg("dj").alias("dj")) \
.orderBy(col("date"), ascending=True).toPandas()


# COMMAND ----------

import altair as alt

st_eda_df.rename(columns={"date":'Date',"sentiment_class":"Sentiment","comment_count":"Daily Comment Count",'avg_len':'Daily Average Text Length','avg_score':'Daily Average Score','sp500':'SP500 Index','nasdaq':"Nasdaq Index",'dj':"Dow Jones Index"},inplace=True)
st_eda_df

# COMMAND ----------

### for the following plots, you can zoom in each plot by draging and selecting a certain region or scrolling the mouse wheel
### filter can be realized by selecting certain region or clicking corresponding bar to filter sentimnet
### each point on the plot represents a day within 2021-01-01 ~ 2022-08-31, you can use tooltip to get detailed information about each point

# COMMAND ----------

### y-axis is daily average score
### x-axis is stock market index
### bar plot shows the sum of the daily comment number after filtering operation interatived with plot

interval = alt.selection_interval()
scales = alt.selection_interval(bind='scales')
scales1 = alt.selection_interval(bind='scales')
click = alt.selection_multi(encodings=['color'])

####### sore
###part1
#sp500
scatter_score_s = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('SP500 Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Score:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Score','SP500 Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).add_selection(
    interval
).add_selection(
    scales
)

hist_score_s = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',
    color='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
).transform_filter(
    interval
)

#nasdap
scatter_score_n = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Nasdaq Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Score:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Score','Nasdaq Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).add_selection(
    interval
).add_selection(
    scales
)

hist_score_n = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
    color='Sentiment'
).transform_filter(
    interval
)

# dj
#nasdap
scatter_score_d = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Dow Jones Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Score:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Score','Dow Jones Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).add_selection(
    interval
).add_selection(
    scales
)

hist_score_d = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',
    color='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
).transform_filter(
    interval
)


###part2
# sp500
scatter_score_s1 = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('SP500 Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Score:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Score','SP500 Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).transform_filter(
    click
).add_selection(
    scales1
)

hist_score_s1 = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
    color=alt.condition(click, 'Sentiment', alt.value('lightgray'))
).add_selection(
    click
)

# nasdaq
scatter_score_n1 = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Nasdaq Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Score:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Score','Nasdaq Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).transform_filter(
    click
).add_selection(
    scales1
)

hist_score_n1 = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
    color=alt.condition(click, 'Sentiment', alt.value('lightgray'))
).add_selection(
    click
)

#dj
scatter_score_d1 = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Dow Jones Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Score:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Score','Dow Jones Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).transform_filter(
    click
).add_selection(
    scales1
)

hist_score_d1 = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
    color=alt.condition(click, 'Sentiment', alt.value('lightgray'))
).add_selection(
    click
)

charts= scatter_score_s & hist_score_s & scatter_score_s1 & hist_score_s1 | scatter_score_n & hist_score_n & scatter_score_n1 & hist_score_n1 | scatter_score_d & hist_score_d & scatter_score_d1 & hist_score_d1

charts = charts.properties(background = '#f3f6f7',
                    title = alt.TitleParams(text = 'Stock Market Index VS Comments Daily Average Score',
                                            subtitle = ['The first row: drag and select region to filter data    |    The second row: filter sentiment by click the corresponding bar'],
                                            font = 'Ubuntu Mono', 
                                            fontSize = 22, 
                                            color = '#696969', 
                                            subtitleFont = 'Ubuntu Mono',
                                            subtitleFontSize = 14, 
                                            subtitleColor = '#3E454F'))

fpath = os.path.join(PLOT_DIR, "sentiment_score.html")
charts.save(fpath)
charts

# COMMAND ----------

### y-axis is daily average text length
### x-axis is stock market index
### bar plot shows the sum of the daily comment number after filtering operation interatived with plot
interval = alt.selection_interval()
scales = alt.selection_interval(bind='scales')
scales1 = alt.selection_interval(bind='scales')
click = alt.selection_multi(encodings=['color'])

####### text length
###part1
#sp500
scatter_len_s = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('SP500 Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Text Length:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Text Length','SP500 Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).add_selection(
    interval
).add_selection(
    scales
)

hist_len_s = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',
    color='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
).transform_filter(
    interval
)

#nasdap
scatter_len_n = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Nasdaq Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Text Length:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Text Length','Nasdaq Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).add_selection(
    interval
).add_selection(
    scales
)

hist_len_n = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',
    color='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
).transform_filter(
    interval
)

# dj
#nasdap
scatter_len_d = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Dow Jones Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Text Length:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Text Length','Dow Jones Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).add_selection(
    interval
).add_selection(
    scales
)

hist_len_d = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',
    color='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
).transform_filter(
    interval
)


###part2
# sp500
scatter_len_s1 = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('SP500 Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Text Length:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Text Length','SP500 Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).transform_filter(
    click
).add_selection(
    scales1
)

hist_len_s1 = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
    color=alt.condition(click, 'Sentiment', alt.value('lightgray'))
).add_selection(
    click
)

# nasdaq
scatter_len_n1 = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Nasdaq Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Text Length:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Text Length','Nasdaq Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).transform_filter(
    click
).add_selection(
    scales1
)

hist_len_n1 = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
    color=alt.condition(click, 'Sentiment', alt.value('lightgray'))
).add_selection(
    click
)

#dj
scatter_len_d1 = alt.Chart(st_eda_df).mark_circle().encode(
     alt.X('Dow Jones Index:Q',
        scale=alt.Scale(zero=False)
    ),
        alt.Y('Daily Average Text Length:Q',
        scale=alt.Scale(zero=False)
    ),tooltip=['Date', 'Sentiment', 'Daily Comment Count', 'Daily Average Text Length','Dow Jones Index'],
    size ='Daily Comment Count:Q',
    color=alt.condition(interval, 'Sentiment:N', alt.value('lightgray'))
).transform_filter(
    click
).add_selection(
    scales1
)

hist_len_d1 = alt.Chart(st_eda_df).mark_bar().encode(
    x='sum(Daily Comment Count):Q',
    y='Sentiment',tooltip=['Sentiment', 'sum(Daily Comment Count)'],
    color=alt.condition(click, 'Sentiment', alt.value('lightgray'))
).add_selection(
    click
)



charts= scatter_len_s & hist_len_s & scatter_len_s1 & hist_len_s1 | scatter_len_n & hist_len_n & scatter_len_n1 & hist_len_n1 | scatter_len_d & hist_len_d & scatter_len_d1 & hist_len_d1

charts = charts.properties(background = '#f3f6f7',
                    title = alt.TitleParams(text = 'Stock Market Index VS Comments Daily Average Text Length',
                                            subtitle = ['The first row: drag and select region to filter data    |    The second row: filter sentiment by click the corresponding bar'],
                                            font = 'Ubuntu Mono', 
                                            fontSize = 22, 
                                            color = '#696969', 
                                            subtitleFont = 'Ubuntu Mono',
                                            subtitleFontSize = 14, 
                                            subtitleColor = '#3E454F'))

fpath = os.path.join(PLOT_DIR, "sentiment_len.html")
charts.save(fpath)
charts

# COMMAND ----------

# save to DBFS
st_result.write.mode('Overwrite').parquet('dbfs:/FileStore/nlp_result.parquet')
