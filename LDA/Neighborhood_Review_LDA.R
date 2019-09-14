
# clear all existing variables
rm(list=ls())

# load the text mining library
library(tm)

# set work directory
setwd("C:\\Yingjie\\Research\\2017PlacePerception\\Exp\\Individual_Review_Doc")

# begin to load data
filenames <- list.files(getwd(),pattern = "*.txt")
files <- lapply(filenames, readLines)
docs <- Corpus(VectorSource(files))

# remove stop words
docs <- tm_map(docs, removeWords, stopwords("en"))
myStopwords <- c("can","also","isn","aren","still","don","however",
                 "always","never","manhattan","will","okay","really",
                 "lot","like","one","get","good","need","better","see",
                 "everyone","area","areas","now","around","much","feel",
                 "every","usually","great","well","away","many","need", "nice",
                 "everything","mostly","best","york","bad","just","live","new",
                 "astoria","island","make","know","two","say",
                 "like","brooklyn","harlem","nyc","queen", "definitely","come","want","use","way","bronx","thing")
docs <- tm_map(docs, removeWords, myStopwords)

# get the document term matrix
dtmInit <- DocumentTermMatrix(docs)
rowTotals <- apply(dtmInit , 1, sum) 
dtm   <- dtmInit[rowTotals> 0, ]  # get the documents whose length is larger than 0 after removing stop words


# here we start to find the suitable K for LDA
library(topicmodels)
library("ldatuning") # use the LDA tuning package

setwd("C:\\Yingjie\\Research\\2017PlacePerception\\Exp")

# here we iterate K from 2 to 10 to find the best K
result <- FindTopicsNumber(
  dtm,
  topics = seq(from = 2, to = 10, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 23000, burnin = 4000, iter=2000, best=TRUE),
  mc.cores = 2L,
  verbose = TRUE
)

# if you want to take a quick peak at the result, use "FindTopicsNumber_plot(result)" 

# save the LDA performances at different Ks into a csv file 
write.csv(result, paste("evaluation_k.csv"))




# create 4 plots based on the 4 metrics 

result<- read.csv("evaluation_k.csv",header = TRUE)


tiff("Griffth.tiff", width=2000,height=1400, res = 300)
mar.default <- c(5,4,4,2) + 0.1
par(mar = mar.default + c(0, 2, 0, 0)) 
plot(result$topics, result$Griffiths2004 ,panel.first = abline(v = 2:10, col = "lightgray", lty = 1), type = "o", pch=15,lty=1, xlim = c(2, 10), col = "black", xlab = "Number of Topics (K)", ylab = "log P(w|K)", lwd = 2, cex=1.5,  cex.lab=1.5, cex.axis=1.5)
dev.off()


tiff("Cao.tiff", width=2000,height=1400, res = 300)
mar.default <- c(5,4,4,2) + 0.1
par(mar = mar.default + c(0, 2, 0, 0)) 
plot(result$topics, result$CaoJuan2009 ,panel.first = abline(v = 2:10, col = "lightgray", lty = 1), type = "o", pch=16,lty=1, xlim = c(2, 10), col = "black", xlab = "Number of Topics (K)", ylab = "Avg Topic Similarity", lwd = 2, cex=1.5,  cex.lab=1.5, cex.axis=1.5)
dev.off()


tiff("Arun.tiff", width=2000,height=1400, res = 300)
mar.default <- c(5,4,4,2) + 0.1
par(mar = mar.default + c(0, 2, 0, 0)) 
plot(result$topics, result$Arun2010 ,panel.first = abline(v = 2:10, col = "lightgray", lty = 1), type = "o", pch=17,lty=1, xlim = c(2, 10), col = "black", xlab = "Number of Topics (K)", ylab = "Symmetric KL Divergence", lwd = 2, cex=1.5,  cex.lab=1.5, cex.axis=1.5)
dev.off()


tiff("Deveaud.tiff", width=2000,height=1400, res = 300)
mar.default <- c(5,4,4,2) + 0.1
par(mar = mar.default + c(0, 2, 0, 0)) 
plot(result$topics, result$Deveaud2014 ,panel.first = abline(v = 2:10, col = "lightgray", lty = 1), type = "o", pch=18,lty=1, xlim = c(2, 10), col = "black", xlab = "Number of Topics (K)", ylab = "JS Divergence", lwd = 2, cex=1.5,  cex.lab=1.5, cex.axis=1.5)
dev.off()

# finish drawing the four plots
# based on the analysis above, we found that k=8 is a suitable number for total topics






# In the following, we do topic modeling for k=8
burnin <- 4000
iter <- 2000
thin <- 500
nstart <- 1 
best <- TRUE
k <- 8   # 3, 7, 9

ldaOut <- LDA(dtm,k,method = "Gibbs", control = list(nstart=nstart, best = best, burnin = burnin, iter = iter, thin = thin))

# Extract the probability of each document in beloning to a topic
ldaOut.topics <- as.matrix(topics(ldaOut))
reviewFilenames <- row.names(ldaOut.topics)
topicProbabilities <- as.data.frame(ldaOut@gamma)
rownames(topicProbabilities) <- reviewFilenames
#write.csv(topicProbabilities, file= paste("LDAGibbs",k,"Doc_Topic_Prob.csv"))

# find the top 20 words of each topic
ldaOut.terms <- as.matrix(terms(ldaOut,20))
termProbabilities <- as.data.frame(ldaOut@beta)

topicTermProb <- NULL
topicTermProbColumnNames <- c()
for(j in 1:k)
{
  topic1Prob <- exp(termProbabilities[j,])
  topic1Prob_trans <- t(topic1Prob)
  topic1Prob_trans_sort<-sort(topic1Prob_trans,decreasing = TRUE)
  
  if(is.null(topicTermProb))
  {
    topicTermProb <- data.frame(ldaOut.terms[,j],topic1Prob_trans_sort[0:20])
  }
  else
  {
    topicTermProb <- data.frame(topicTermProb,ldaOut.terms[,j],topic1Prob_trans_sort[0:20])
  }
  topicTermProbColumnNames <- c(topicTermProbColumnNames,paste("Topic",j), paste("Topic",j,"Prob"))
}
colnames(topicTermProb) <- topicTermProbColumnNames
write.csv(topicTermProb, file= paste("LDAGibbs",k,"Topic_Term_Prob.csv"))



