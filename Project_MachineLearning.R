################################################################
## Project_MachineLearning.R - This program contains exploratory analysis, pre-processing ##
## and model fitting (plus evaluation) for the "Weight Lifting Form-Correctness" project. ##
## Author: Ganesh Krishnamurthy ##
## Creation date: 07/20/2015 ##
################################################################

# Load the necessary libraries
library(caret); library(ggplot2)

# Set the working directory
setwd("C:/Training/R_Programming/Machine_Learning/Project")

# Read-in the training dataset. Read in blanks as NA using the na.strings argument
training<-read.csv("pml-training.csv", header=TRUE, na.strings=c("","NA"))

# High level exploration of data
dim(training) # 19622 records and 160 columns
class(training$classe)

################################################################################################################
# Pre-process the data frame to identify and remove non-sensical (too many nulls, no variablity etc) predictors##
################################################################################################################

# Remove the records with new_window="yes" and then check. This is necessary since for many otherwise null
# columns, for the new_windwo="yes" record, there are cumulative values like skewness, kurtosis, stddev etc for
# the previous window of observations. 
training_exWindow<-subset(training, new_window=="no") # 19216 records. So, 406 records removed

# Identify zero covariates
nsv<-nearZeroVar(training_exWindow,saveMetrics = TRUE)

# Check how many variables were "near zero" vs "zero"
table(nsv$zeroVar, nsv$nzv) # Result: 101 bad predictors and 59 good ones.
# The zero var and near zero var classifications are mirror images

# Create a new dataset that doesn't have the non-sensical predictors with NA values. Use the nsv vector created
# above in choosing the columns to include
training_clean<-training[,nsv$zeroVar==FALSE] # 59 columns vs 160 columns in the initial set

# Further clean up dataframe (predictors) by removing the first 6 columns (row number, user name, time stamp etc)
# don't provide any predictive value. They come in the way of a model formula: classe ~ .
training_clean<-training_clean[,-(1:6)]
dim(training_clean) # 19622 rows and 53 columns (52 predictors and 1 classe dependent variable)

###############################################################
## Dimension reduction: Check for correlation between predictors and apply ##
## principal components if necessary ##
###############################################################
Cor_Mat<-abs(cor(training_clean[,-53]))
diag(Cor_Mat)<-0 # Remove the diagonal (all 1)
summary(Cor_Mat[upper.tri(Cor_Mat)])
corr_preds<-which(Cor_Mat > 0.8, arr.ind=TRUE) # Store the correlated predictors
# Result: 38 predictors are correlated 

# Now try principal components and use scale=TRUE (advised)
pc_comp<-prcomp(training_clean[,-53], scale.=TRUE)
summary(pc_comp) ## It now takes 12 Principal components to explain 80 percent variation

# Plot the PCs
plot(pc_comp$x[,1],pc_comp$x[,2], col=training_clean$classe)
# Result: There are 5 distinct blobs in the plot. But the blobs have all the classes

# Create PCA dataframe containing the 12 PCAs (accounting for 80 percent variation) #
# and attach the "classe" column to it. This will become the training data 
# in subsequent steps
pcaObj<-preProcess(training_clean[,-53], method="pca", thresh=0.8)
training_PC<-predict(pcaObj, training_clean[,-53])
training_PC<-data.frame(training_PC, classe=training_clean$classe) # Add back the target/classe 

###############################################################
## Create a "trainControl" function to implement 10 fold cross validation. This will be used ##
## as parameter to the model fit/"train" function in models fitted below. ##
###############################################################
train_control_10k_CV<-trainControl(method="cv", number=10)

#####################################################################################
## Model 1: TREE on raw data. Use 10-fold cross validation on entire training data ##
#####################################################################################
modFit_tree_raw<-train(classe ~ ., method="rpart", data=training_clean, trControl=train_control_10k_CV)

# Classification accuracy
prop.table(table(training_clean$classe, predict(modFit_tree_raw)),1)
confusionMatrix(training_clean$classe, predict(modFit_tree_raw, training_clean[,-53]))
# Result: 49.56% accuracy

#####################################################
## Model 2: TREE on centered and scaled data. Use 10-fold cross validation ##
#####################################################
modFit_tree_scaled<-train(classe ~ ., method="rpart", data=training_clean,
                          trControl=train_control_10k_CV, 
                          preProcess=c("center","scale"))

# Classification accuracy
prop.table(table(training_clean$classe, predict(modFit_tree_scaled)),1)
confusionMatrix(training_clean$classe, predict(modFit_tree_scaled, training_clean[,-53]))
# Result: 49.56% accuracy (no change from unscaled fit)

######################################################
## Model 3: Random Forest ##
## A. Fit on principal components WITHOUT cross validation - main purpose  ##
## is to "learn" what causes mempry problems etc, how many predictors are best etc ##
######################################################

modFit_randForest<-train(classe ~ ., method="rf", data=training_PC)

# Classification accuracy
confusionMatrix(training_PC$classe, predict(modFit_randForest, training_PC[,-13]))
# Result: 100% accuracy on the data that trained the model

## B. Fit the random Forest with 10 fold cross validation
modFit_randForest_10cv<-train(classe ~ ., method="rf", data=training_PC, trControl=train_control_10k_CV)
# Classification accuracy
confusionMatrix(training_PC$classe, predict(modFit_randForest_10cv, training_PC[,-13]))
prop.table(table(training_PC$classe, predict(modFit_randForest_10cv, training_PC[,-13])),1)
# Result: Again, 100% accuracy. Too go to be true!!!

modFit_randForest_10cv$finalModel

########################################################################
## 100% Accuracy is alarming: ##
## Take out a hold out sample of 10 percent and train the model on the 90% with 10k-k
## CV. Then, check if the 100% accuracy results still happens on the hold out sampple
########################################################################
inTrain<-createDataPartition(y=training_clean$classe, p=.9, list=FALSE)
train_90<-training_clean[inTrain,]
test_10<-training_clean[-inTrain,]
dim(train_90); dim(test_10)

train_pcaObj<-preProcess(train_90[,-53], method="pca", thresh=0.8)
train_90_PC<-predict(train_pcaObj, train_90[,-53])
train_90_PC<-data.frame(train_90_PC, classe=train_90$classe)

modFit_randForest_tr90<-train(classe ~ ., data=train_90_PC, method="rf", trControl=train_control_10k_CV)
# Create the PCA processed test dataset
test_10_PC<-predict(train_pcaObj,test_10[,-53])
# Get accuracy on this 10% test dataset
confusionMatrix(test_10$classe, predict(modFit_randForest_tr90, test_10_PC))
# Result: 97.6% (almost 100%) on the hold out 10% sample. This is SUPER!

######################################################################
## Model 4: BOOSTING model ##
######################################################################
modFit_boost<-train(classe ~ ., data=training_PC, method="gbm", verbose=FALSE,
                    trControl=train_control_10k_CV)

# Classification accuracy
confusionMatrix(training_PC$classe, predict(modFit_boost, training_PC[,-13]))
# Result: 78.49% accuracy
prop.table(table(training_PC$classe, predict(modFit_boost, training_PC[,-13])),1)

######################################################################
## Model 5: Linear Discriminant Analysis with 10-fold CV ##
######################################################################
modFit_lda<-train(classe ~., data=training_PC, method="lda", trControl=train_control_10k_CV)

# Classification accuracy
confusionMatrix(training_PC$classe, predict(modFit_lda, training_PC[,-13]))
# Result: 46.7% accuracy. Pretty bad compared to the random forest and boosted models

######################################################################
## Scoring the TEST dataset: Read the test dataset (in .csv format) and use the best model ##
## i.e., the Random Forest to make the predictions. ##
######################################################################
testing<-read.csv("pml-testing.csv", header=TRUE, na.strings=c("","NA"))

# Apply the same column clean (removal) done on training dataset
testing_clean<-testing[,nsv$zeroVar==FALSE]
testing_clean<-testing_clean[,-59] # Drop the problem ID column as it was not in the training dataset
testing_clean<-testing_clean[,-(1:6)] # Drop the time stamp, user name and window number 

## Create principal components using the pcaObj object trained on the training dataset
testing_PC<-predict(pcaObj, testing_clean)

## Predict using the randomForest
testing_predictions<-predict(modFit_randForest_10cv, newdata=testing_PC) ## A factor vector is returned
## Convert predictions to a character vector
testing_predictions_char<-as.character(testing_predictions)

## Create 20 files with the answers (each in 1 file). Using the pml_write_files() function #
## provided by instructor Jeffrey Leek
setwd("./Submissions")

pml_write_files<-function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testing_predictions_char)
