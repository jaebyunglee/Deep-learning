rm(list=ls())
######################data###############################33333
train.data = read.csv("C:/Users/kis91/Desktop/python study/data/titanic/train.csv")
train.data = train.data[,-1]
test.data = read.csv("C:/Users/kis91/Desktop/python study/data/titanic/test.csv")
test.data = test.data[,-1]
test.data = data.frame(NA,test.data)
colnames(test.data) = colnames(train.data)
data = rbind(train.data,test.data)
data = subset(data,select = -c(Name,Ticket))
data$Pclass = as.factor(data$Pclass)
levels(data$Cabin) = c(levels(data$Cabin),"None")
data$Cabin[data$Cabin==""] = "None"
data$Cabin = as.factor(substr(data$Cabin,1,1))
levels(data$Embarked) = c(levels(data$Embarked),"None")
data$Embarked[data$Embarked==""] = "None"
data[is.na(data$Fare),]$Fare = mean(data[!is.na(data$Fare),]$Fare)


#NA Age predict
age.train = data[!is.na(data$Age),-1]
age.test = data[is.na(data$Age),-1]
age.fit = lm(Age~.,data=age.train)
data[is.na(data$Age),]$Age = predict(age.fit,newdata = age.test)

#scale numeric variables
data[,c("Age","SibSp","Parch","Fare")] = apply(data[,c("Age","SibSp","Parch","Fare")],2,scale)

#train, test data
train = data[!is.na(data$Survived),]
test = data[is.na(data$Survived),]
test$Survived = 1

############################## neural network with keras #############################
library(keras)
library(caret)
library(Matrix)
library(xlsx)

train.x = sparse.model.matrix(Survived ~ .-1, data = train)
train.y = train[,1]
test.x = sparse.model.matrix(Survived ~ .-1, data = test)




# #callbacks
# callbacks_list = list(
#   callback_early_stopping(
#     monitor = "acc",
#     patience = 1)
# )

folds = 5
num_epochs = 100
set.seed(2019)
cvIndex = createFolds(train.y,5)
ee.mat = NULL

###tuning parameters
tune.par = expand.grid("units" = c(16,64),
            "rate" = c(0.3,0.5),
            "regularizer" = c(0.01,0.001))

###grid search
for(j in 1:nrow(tune.par)){
  cv.err.mat = NULL
  # 5 fold cross validation
  for(i in 1:folds){
    cat('#tune :',j,'#fold :',i,'\n')
    val_data = train.x[cvIndex[[i]],]
    val_target = train.y[cvIndex[[i]]]
    
    partial_train_data = train.x[-cvIndex[[i]],]
    partial_train_target = train.y[-cvIndex[[i]]]
    
    #build model
    model = keras_model_sequential() %>% 
      layer_dense(units = tune.par[j,"units"], 
                  kernel_regularizer = regularizer_l2(tune.par[j,"regularizer"]), 
                  activation = "relu", input_shape = dim(train.x)[2]) %>% #shape is demension
      layer_dropout(rate = tune.par[j,"rate"]) %>% 
      layer_dense(unit = tune.par[j,"units"], 
                  kernel_regularizer = regularizer_l2(tune.par[j,"regularizer"]), 
                  activation = "relu") %>% 
      layer_dropout(rate = tune.par[j,"rate"]) %>% 
      layer_dense(units = 1, activation = "sigmoid")
    
    #compile model
    model %>% compile(
      optimizer = "rmsprop",
      loss = "binary_crossentropy",
      metrics = c("accuracy")
    )
    
    #save each epochs accuracy
    history = model %>% fit(partial_train_data,partial_train_target,
                            validation_data = list(val_data,val_target), #val data
                            epochs = num_epochs, batchsize = 24, verbose = 0)
    
    cv.err.mat = rbind(cv.err.mat,history$metrics$val_acc)
  }
  opt.epochs = which.max(colMeans(cv.err.mat))
  ee.mat = rbind(ee.mat,c(opt.epochs,colMeans(cv.err.mat)[opt.epochs] ))
}
colnames(ee.mat) = c("epochs","acc")
final.tune.mat = cbind(tune.par,ee.mat)
opt.par = final.tune.mat[which.max(final.tune.mat$acc),]

###build model with optmal parameters
model = keras_model_sequential() %>% 
  layer_dense(units = opt.par$unit, 
              kernel_regularizer = regularizer_l2(opt.par$regularizer), 
              activation = "relu", input_shape = dim(train.x)[2]) %>% 
  layer_dropout(rate = opt.par$rate) %>% 
  layer_dense(unit = opt.par$unit, 
              kernel_regularizer = regularizer_l2(opt.par$regularizer), 
              activation = "relu") %>% 
  layer_dropout(rate = opt.par$rate) %>% 
  layer_dense(units = 1, activation = "sigmoid")

###compile model
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

### fit model with optimal tuning parameters
model %>% fit(train.x,train.y,
              validation_data = list(val_data,val_target), #val data
              epochs = opt.par$epochs, batchsize = 24, verbose = 0)

final.value = (predict(model,test.x)>0.5)+0

write.xlsx(final.value, file="C:/Users/kis91/Desktop/submission1.xlsx")
