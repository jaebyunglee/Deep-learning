rm(list=ls())

library(keras)
library(ggplot2)
#data - under graduate Data mining
a.train_data = read.csv("C:\\Users\\kis91\\Desktop\\old.sam.for.reg.fit.csv",header = T)
a.test_data = read.csv("C:\\Users\\kis91\\Desktop\\old.sam.for.reg.pred.csv",header = T)

train_targets = a.train_data[,1]
train_data = a.train_data[,-1]
test_targets = a.test_data[,1]
test_data = a.test_data[,-1]

mean = apply(train_data[,-c(1,2)],2,mean)
std = apply(train_data[,-c(1,2)],2,sd)
train_data = scale(train_data[,-c(1,2)],center = mean, scale = std)
test_data = scale(test_data[,-c(1,2)],center = mean, scale = std)
train_data = as.matrix(train_data)
test_data = as.matrix(test_data)
##########################################################
build_model = function(){
  model = keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

############################################
#3.24
#k-folds cross validation
k = 4
indices = sample(1:nrow(train_data))
folds = cut(1:length(indices), breaks = k, labels = F)

num_epochs = 50
all_scores = c()

for(i in 1:k){
  cat("processing fold #", i, "\n")
  
  val_indices = which(folds == i, arr.ind = TRUE)
  val_data = train_data[val_indices,]
  val_targets = train_targets[val_indices]
  
  partial_train_data = train_data[-val_indices,]
  partial_train_target = train_targets[-val_indices]
  
  model = build_model()
  model %>%  fit(partial_train_data,partial_train_target,
                 epochs = num_epochs,batch_size =1, verbose = 0)
  results = model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores = c(all_scores,results$mean_absolute_error)
}
all_scores
mean(all_scores)

#3.25
num_epochs = 50
all_mae_histories = NULL
for(i in 1:k){
  cat("processing fold #", i, "\n")
  
  val_indices = which(folds == i, arr.ind = TRUE)
  val_data = train_data[val_indices,]
  val_targets = train_targets[val_indices]
  
  partial_train_data = train_data[-val_indices,]
  partial_train_target = train_targets[-val_indices]
  
  model = build_model()
  history = model %>% fit(
    partial_train_data,partial_train_target,
    validation_data = list(val_data,val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  mae_history = history$metrics$val_mean_absolute_error
  all_mae_histories = rbind(all_mae_histories,mae_history)
}

average_mae_history = data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories,2,mean)
)

ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

# 3.29 final model
model = build_model()
model %>% fit(train_data,train_targets,epochs = 7, batch_size = 16, verbose = 0)
results = model %>% evaluate(test_data,test_targets)
results
prediction = model %>% predict(test_data)
# deep learning mse
dl.mae = sum(abs(test_targets-drop(prediction)))/length(test_targets)

###############################linear regression ###################################
#stepwise regression
#data
train_data
xy.df = data.frame(cbind(train_targets,train_data))
fit.lm = lm(train_targets~. ,data = xy.df)
summary(fit.lm)
beta = step(fit.lm,direction = "both")
names(beta$coefficients)[-1]
#regression mse
st.mae = sum(abs(test_targets - drop(cbind(1,test_data[,names(beta$coefficients)[-1]])%*%beta$coefficients)))/length(test_targets)
################################ lasso ###################################
#lasso
library(glmnet)
train_targets = as.vector(train_targets)
fit.lasso = cv.glmnet(train_data,train_targets)
pos.lasso = fit.lasso$lambda==fit.lasso$lambda.min
beta.lasso = rbind(fit.lasso$glmnet.fit$a0,fit.lasso$glmnet.fit$beta)[,pos.lasso]
#lasso mse
la.mae = sum(abs(test_targets - drop(cbind(1,test_data)%*%beta.lasso)))/length(test_targets)
################################ SCAD ###################################
library(ncvreg)
train_targets = as.vector(train_targets)
fit.SCAD = cv.ncvreg(train_data,train_targets)
pos.SCAD = fit.SCAD$lambda==fit.SCAD$lambda.min
beta.SCAD = fit.SCAD$fit$beta[,pos.SCAD]
#SCAD mse
sc.mae = sum(abs(test_targets - drop(cbind(1,test_data)%*%beta.SCAD)))/length(test_targets)
############################## robust regression #######################
library(robustreg)

seq.vec = seq(0.1,1,length.out = 100)
e.vec = rep(0,100)
a.mat = matrix(NA,100,19)
for(j in 1:100){
  fit.rob = robustRegH(train_targets~. ,data = xy.df,tune = seq.vec[j])
  a.mat[j,] = as.vector(fit.rob$coefficients)
  e.vec[j] = sum(abs(test_targets - drop(cbind(1,test_data)%*%a.mat[j,])))/length(test_targets)
}

#robust mse
rb.mae = e.vec[which.min(e.vec)]

mae = cbind(dl.mae,st.mae,la.mae,sc.mae,rb.mae)
mae


