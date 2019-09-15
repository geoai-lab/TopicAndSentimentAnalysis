# Semantic and Sentiment Analysis on Neighborhood Reviews

* Author: Yingjie Hu
* Email: yjhu.geo@gmail.com


### Overall description 
This project provides the code for performing semantic and sentiment analysis on online neighborhood reviews. The goal is to understand the perceptions and feelings of people toward their neighborhoods. The neighborhood review data used in this project are retrieved from the website Niche: https://www.niche.com/. We perform semantic analysis to understand the aspects that people talk about their neighborhoods, and perform sentiment analysis to understand the emotions and attitudes of people toward these aspects. More details about this project can be found in the following paper:

Hu, Y., Deng, C. and Zhou, Z., 2019. A semantic and sentiment analysis on online neighborhood reviews for understanding the perceptions of people toward their living environment. Annals of the American Association of Geographers, 109(4), 1052-1073. http://www.acsu.buffalo.edu/~yhu42/papers/2019_Annals_NeighborhoodReview.pdf

Please feel free to re-use the code here for your own projects. Niche does not allow publicly sharing their data, so no raw data is included in this repository. If you have any questions, please feel free to contact the first author. Note that the LDA R code is written on Windows while all the other codes (in Java or Python) are written and tested on Linux (Ubuntu). There can be path format issues if you use the programs on windows (windows use "\\" while linux use "/"). Please change the format of file paths accordingly.

A figure with a screenshot of the neighborhood reviews on Niche and the average review ratings of NYC neighborhoods is shown as below:
<p align="center">
<img align="center" src="https://github.com/geoai-lab/TopicAndSentimentAnalysis/blob/master/fig/NeighborhoodReview.png" width="600" />
</p>


### Repository organization


* "LDA": This folder contains the R code for finding the suitable K value for LDA (LDA tuning) and for finding the frequent words for each topic when k equals 3, 7, 8, 9. 

* "MGLDA": This folder contains the Python code for MGLDA. In order to run this code, you will need the Python package "nltk". After nltk is installed, you will also need to do "nltk.download('stopwords')" and "nltk.download('wordnet')" in command lines.

* "LARA": This folder contains the Java code for running LARA which decomposes each review into aspect-specific scores based on the topics identified. It will need Stanford NLP library and additional libraries in the folder. It functions in two steps: (1) Run "NeighborhoodAnalyzer.java" to first identify words related to each aspect using bootstrapping and then decompose each review into sub aspects. (2) Run "LRR.java" to obtain the decomposed rating, which will be saved into the "combinedResult.csv" file.

* "WordCloud": This folder contains the Python code for generating word clouds based on the frequent words of LDA topics.

* "ReviewDataExploreStatistics": This folder contains the Python code for performing an exploratory analysis on the lengths of the neighborhood reviews and for generating a histogram.

* "NaiveAspectSentiment": This folder contains the Python code for generating aspect-specific ratings of reviews based on two simple approaches: the naive approach and the sentiment word based approach.

* "NeighborhoodMTurk": This folder contains the html, javascript, and css code for performing the Amazon Mechanical Turk experiment. In the experiments, AMT users are asked to annotate the topics of each review and choose suitable scores for the aspects.

* "Sentiment_Evaluation": This folder contains the Python code for comparing the performances of LDA_LARA, naive, and sentiment-word approach in decomposing neighborhood reviews.




