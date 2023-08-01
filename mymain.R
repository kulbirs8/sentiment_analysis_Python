#####################################
# Load required Libraries
#####################################

library(kableExtra)
library(text2vec)
library(tokenizers)
library(xgboost)
library(pROC)

#####################################
# Loading vocabulary and training and Test data
#####################################

getwd()

setwd("C:/Users/kulsk/Desktop/mcsds/cs 598 psl/4/Project 3")

myvocab <- scan(file = "myVocab.txt", what = character())


train <- read.table("train.tsv", stringsAsFactors = FALSE,
                    header = TRUE)

test <- read.table("test.tsv", stringsAsFactors = FALSE,
                   header = TRUE)

#####################################
#
# Train a binary classification model
#
#####################################




Compute_prediction <- function(myvocab, train, test,tokenizer = word_tokenizer)
  
{
  
  i_train = itoken(train$review, 
                   preprocessor = tolower, 
                   tokenizer = word_tokenizer, 
                   ids = train$id, 
                   progressbar = FALSE)
  
  i_test = itoken(test$review, 
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer, 
                  ids = test$id, 
                  progressbar = FALSE)
  
  
  voc= create_vocabulary(myvocab, ngram = c(1L, 3L))
  
  
  vectorizer  = vocab_vectorizer(voc)
  
  dtm_train  = create_dtm(i_train, vectorizer)
  dtm_test = create_dtm(i_test, vectorizer)
  
  
  preds = model_pre(dtm_train, train$sentiment, dtm_test)
  return (preds)
}


model_pre <- function(train, label, test) 
{
  binary_classification_model = xgboost(data = train, 
                                        label=label,
                                        objective = "binary:logistic", 
                                        eval_metric = "auc",
                                        eta = 0.09,
                                        nrounds = 3000)
  
  return (predict(binary_classification_model, test, type="response"))
}


set.seed(1008)
prediction = Compute_prediction(myvocab, train, test)
output = cbind(id = test$id, prob = round(prediction,2))


#####################################
# Compute prediction 
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predited probabilities
#####################################

write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')




