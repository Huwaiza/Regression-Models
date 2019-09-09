# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

#Encoding categorical data
dataset$State = factor(dataset$State,
                      level = c('New York','California','Florida'),
                      labels = c(1,2,3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)

split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#multiple linear regression
#regressor = lm(formula = Profit ~  R.D.Spend + Administration + Marketing.Spend + S.)

#R tells you about everything the p values and prediction importance

#regressor = lm(formula = Profit ~  . ,
#               data = training_set)

#Since R,D.Spend has 3 starts the highest statistical importance we can write like this
regressor = lm(formula = Profit ~  R.D.Spend ,
               data = training_set)
#predicting the test set results
y_pred = predict(regressor , newdata = test_set)

#backward elimination
regressor = lm(formula = Profit ~  R.D.Spend + Administration + Marketing.Spend + State,
                 data = dataset)
summary(regressor)

#backward elimination
regressor = lm(formula = Profit ~  R.D.Spend + Administration + Marketing.Spend ,
               data = dataset)
summary(regressor)


#backward elimination
regressor = lm(formula = Profit ~  R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

#backward elimination
regressor = lm(formula = Profit ~  R.D.Spend,
               data = dataset)
summary(regressor)

#predicting the test set results
y_pred = predict(regressor , newdata = training_set)
