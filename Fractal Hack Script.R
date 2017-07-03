#Loading required libraries
library(caret)
library(xgboost)
library(dplyr)
library(ggplot2)


#Reading train n test
train<-read.csv("train.csv")
test<-read.csv("test.csv")

sample_sub<-read.csv("Sample_Submission_Zxs5Ys1.csv")


#Correcting data types
#train$Category_1<-as.factor(train$Category_1)
#test$Category_1<-as.factor(test$Category_1)

train$Category_1<-as.numeric(train$Category_1)
test$Category_1<-as.numeric(test$Category_1)

train$Category_2<-as.numeric(train$Category_2)
test$Category_2<-as.numeric(test$Category_2)

train$Category_3<-as.factor(train$Category_3)
test$Category_3<-as.factor(test$Category_3)

train$Datetime<-as.Date(train$Datetime,"%Y-%m-%d")
test$Datetime<-as.Date(test$Datetime,"%Y-%m-%d")


#Creating features

train$year<-as.numeric(format(train$Datetime, "%Y"))
test$year<-as.numeric(format(test$Datetime, "%Y"))

train$month<-as.numeric(format(train$Datetime, "%m"))
test$month<-as.numeric(format(test$Datetime, "%m"))

train$day<-as.numeric(format(train$Datetime, "%d"))
test$day<-as.numeric(format(test$Datetime, "%d"))

train$weekday<-as.factor(weekdays(train$Datetime))
test$weekday<-as.factor(weekdays(test$Datetime))


train_features<-as.data.frame(train %>% 
                                group_by(Item_ID) %>% 
                                summarise(sum_price_n = sum(Price)/ log(1+length(Price)), mean_price = mean(Price), median_price = median(Price), sd_price = sd(Price), nprice = length(Price),
                                          nsales_price_n = sum(Number_Of_Sales)/ log(1+length(Number_Of_Sales)), mean_nsales = mean(Number_Of_Sales), median_nsales = median(Number_Of_Sales), sd_nsales = sd(Number_Of_Sales), nsales_id = length(Number_Of_Sales)))

test_features<-as.data.frame(test %>% 
                                group_by(Item_ID) %>% 
                               summarise(nprice_2 = length(ID),
                                          nsales_id_2 = length(ID)))


train_new<-merge(train,train_features,by = "Item_ID")

test_new<-merge(test,train_features,by = "Item_ID")

train_new2<-merge(train_new,test_features,by = "Item_ID")

test_new2<-merge(test_new,test_features,by = "Item_ID")


train_new2$diff_p<-train_new2$nprice-train_new2$nprice_2
train_new2$diff_s<-train_new2$nsales_id-train_new2$nsales_id_2

diff_df<-as.data.frame(train_new2[,c("Item_ID", "diff_p", "diff_s")] %>% 
                         group_by(Item_ID) %>% 
                         summarise(diff_p = mean(diff_p),
                                   diff_s = length(diff_s)))


test_new3<-merge(test_new2,diff_df,by = "Item_ID")


train_new2$Datetime<-NULL
test_new3$Datetime<-NULL

y_price<-train_new2$Price
y_nsales<-train_new2$Number_Of_Sales

y_price_l<-log(y_price+1)
y_nsales_l<-log(y_nsales+1)

train_new2$Price<-NULL
train_new2$Number_Of_Sales<-NULL

test_item_id<-test_new3$Item_ID
test_id<-test_new3$ID

#train_new2$mean_price_max <- train_new2$mean_price+train_new2$sd_price
#train_new2$mean_nsales_max <- train_new2$mean_nsales+train_new2$sd_nsales

#train_new2$mean_price_max <- train_new2$mean_price-train_new2$sd_price
#train_new2$mean_nsales_max <- train_new2$mean_nsales-train_new2$sd_nsales

train_new2$meansd_price <- train_new2$mean_price*train_new2$sd_price
train_new2$meansd_nsales <- train_new2$mean_nsales*train_new2$sd_nsales


test_new3$meansd_price <- test_new3$mean_price*test_new3$sd_price
test_new3$meansd_nsales <- test_new3$mean_nsales*test_new3$sd_nsales

train_new2$Item_ID<-NULL
test_new3$Item_ID<-NULL

train_new2$ID<-NULL
test_new3$ID<-NULL

plot(train_new2$sum_price_n,y_price_l)

plot(train_new2$mean_price,y_price_l)

plot(train_new2$nprice,y_price_l)



dmy <- dummyVars(" ~ .", data = train_new2, fullRank = F)
x_train <- data.frame(predict(dmy, newdata = train_new2))

dmy <- dummyVars(" ~ .", data = test_new3, fullRank = F)
x_test <- data.frame(predict(dmy, newdata = test_new3))

#x_train$is_weekend<-as.numeric(ifelse(x_train$weekday.Saturday==1|x_train$weekday.Sunday==1,1,0))
#x_test$is_weekend<-as.numeric(ifelse(x_test$weekday.Saturday==1|x_test$weekday.Sunday==1,1,0))


x_train$mean_price_l<-log(1+x_train$mean_price)
x_train$mean_nsales_l<-log(1+x_train$mean_nsales)

x_test$mean_price_l<-log(1+x_test$mean_price)
x_test$mean_nsales_l<-log(1+x_test$mean_nsales)

x_train$sd_price_l<-log(1+x_train$sd_price)
x_train$sd_nsales_l<-log(1+x_train$sd_nsales)

x_test$sd_price_l<-log(1+x_test$sd_price)
x_test$sd_nsales_l<-log(1+x_test$sd_nsales)



predictors<-c("Category_3.0", "Category_3.1", "Category_2", "Category_1", 
              "year", "month", "day", "weekday.Friday", "weekday.Monday", "weekday.Saturday", 
              "weekday.Sunday", "weekday.Thursday", "weekday.Tuesday", "weekday.Wednesday", 
              "mean_price_l", "mean_nsales_l", "sd_price_l", "sd_nsales_l")


predictors<-c("Category_3.1", "Category_2", "Category_1", 
              "year", "month", 
              "mean_price_l", "mean_nsales_l", "sd_price_l", "sd_nsales_l")


dtrain_price <- xgb.DMatrix(data=as.matrix(x_train[,predictors]), label=y_price_l)
dtrain_sales <- xgb.DMatrix(data=as.matrix(x_train[,predictors]), label=y_nsales_l)

dtest <- xgb.DMatrix(data=as.matrix(x_test[,predictors]))




#watchlist <- list(train=dtrain_un2, validation=dval)

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  precision <- sum(preds & labels) / sum(preds)
  recall <- sum(preds & labels) / sum(labels)
  fmeasure <- 2 * precision * recall / (precision + recall)
  return(list(metric = "F-Score", value = fmeasure))
}

param <- list(
  objective = "reg:linear",
  eval_metric = "rmse",  
  eta = 0.3,
#  lambda = 2
  max_depth = 2
)

cv_price <- xgb.cv(
  params = param, 
  data = dtrain_price, 
  nrounds = 10000, 
#  watchlist = watchlist,
  nfold = 10,  
  maximize = F,
  early_stopping_rounds = 20,
  print_every_n = 50
)

xgb_price <- xgb.train(
  params = param, 
  data = dtrain_price, 
  nrounds = 1500
#  watchlist = watchlist,
#  maximize = F,
#  early_stopping_rounds = 20
)




param2 <- list(
  objective = "reg:linear",
  eval_metric = "rmse",  
  eta = 0.3,
  #  lambda = 2
  max_depth = 2
)



set.seed(1)

cv_sales <- xgb.cv(
  params = param2, 
  data = dtrain_sales,
  nrounds = 10000, 
  nfold = 10,
  print_every_n  = 20
)


xgb_nsales <- xgb.train(
  params = param2, 
  data = dtrain_sales, 
  nrounds = 900, 
  #  watchlist = watchlist,
  maximize = T
  #  early_stopping_rounds = 20
)



gg <- xgb.ggplot.importance(xgb.importance(model = xgb_price,feature_names =  predictors), measure = "Gain", rel_to_first = TRUE)
gg + ggplot2::ylab("Frequency")

gg <- xgb.ggplot.importance(xgb.importance(model = xgb_nsales,feature_names =  predictors), measure = "Gain", rel_to_first = TRUE)
gg + ggplot2::ylab("Frequency")


y_pred_price<-exp(predict(object = xgb_price,newdata = dtest))-1
y_pred_nsales<-exp(predict(object = xgb_nsales,newdata = dtest))-1

y_pred_price[y_pred_price<0]<-0
y_pred_nsales[y_pred_nsales<0]<-0

sub<-data.frame(ID = test_id, Number_Of_Sales = y_pred_nsales, Price = y_pred_price)

#Final submission
#Public LB Rank: 5
#Private LB Rank: 4 
write.csv(sub, "sub16_without_weekdays.csv", row.names = F)



#Linear model

tr<-x_train[,predictors]
te<-x_test[,predictors]


for(i in 1:ncol(tr))
{
  if(is.numeric(tr[,i]))
  {
    tr[is.na(tr[,i]),i]<-as.numeric(median(tr[,i], na.rm = TRUE))
  }
  
}

for(i in 1:ncol(te))
{
  if(is.numeric(te[,i]))
  {
    te[is.na(te[,i]),i]<-as.numeric(median(te[,i], na.rm = TRUE))
  }
  
}



tr$y_price_l<-y_price_l
tr$y_nsales_l<-y_nsales_l

model_price = lm(y_price_l ~ ., data = tr[,predictors])
model_sales = lm(y_nsales_l ~ ., data = tr[,predictors])

y_pred_price<-exp(predict(object = model_price,newdata = te))-1
y_pred_nsales<-exp(predict(object = model_sales,newdata = te))-1

y_pred_price[y_pred_price<0]<-0
y_pred_nsales[y_pred_nsales<0]<-0

sub<-data.frame(ID = test_id, Number_Of_Sales = y_pred_nsales, Price = y_pred_price)

#Did not used the linear model
#write.csv(sub, "sub14_with_mean_sd_both_log_plus_one_linear.csv", row.names = F)
