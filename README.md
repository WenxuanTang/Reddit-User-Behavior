# Reddit User Behavior

## Project overview and objective

I worked with the [Reddit Archive data](https://files.pushshift.io/reddit/) from January 2021 through the end of August 2022, representing about 8TB of uncompressed text JSON data which have been converted to two parquet files (about 1TB). For a sneak peek of the data, you can download sample files of the submissions and comments:

* [Submissions sample JSON](https://files.pushshift.io/reddit/submissions/sample.json)
* [Comments sample JSON](https://files.pushshift.io/reddit/comments/sample_data.json)

I used Spark on Azure Databricks and Python to analyze, transform, and process your data and create one or more analytical datasets.

In this work as data scientists, in addition to doing modeling and machine learning work, I also provided the following as part of a project:

* **Findings:** what does the data say?
* **Conclusions:** what is the interpretation of the data?
* **Recommendations:** what can be done to address the question/problem at hand?


## Milestones

The project executed over several milestones.There are four major milestones:

1. [Milestone 1: Define the questions and Exploratory Data Analysis]
2. [Milestone 2: NLP and external data overlay]
3. [Milestone 3: Machine Learning]
4. [Milestone 4: Final delivery]

Here's the [website](https://wenxuantang.github.io/Reddit-User-Behavior/) I built to host the project.


## Repository structure

 The repository has the following structure:

```.
├── LICENSE
├── README.md
├── code/
├── data/
├── website/
└── website-source/
```
### Description

* The `code/` directory is where I wrote all of your scripts. I had a combination of Pyspark and Python notebooks, and one sub-directory per major task area. 
* The `data/` directory contains.
* The `website/` directory where the final website will be built. 
* The `website-source/` is where I developed the website using preferred method. It rendered in `website/`.
