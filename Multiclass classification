rm(list=ls())

library(keras)
#data
reuters = dataset_reuters(num_words = 10000)
c(c(train_data,train_labels),c(test_data,test_labels)) %<-% reuters

#data convertion
vec_seq.fun = function(data,dim=10000){
  results = matrix(0,length(data),dim)
  for(i in 1:length(data)){
    results[i,data[[i]]] = 1
  }
  return(results)
}

x_train = vec_seq.fun(train_data,dim=10000)
x_test = vec_seq.fun(test_data,dim=10000)

to_one_hot.fun = function(data,dim=46){
  results = matrix(0,length(data),46)
  for(i in 1:length(data)){
    results[i,data[[i]]+1] = 1
  }
  return(results)
}

y_train = to_one_hot.fun(train_labels,dim=46)
y_test = to_one_hot.fun(test_labels,dim=46)
#############################################################################################
#model
model = keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

#compile
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#data split
val_indices = 1:1000
x_val = x_train[val_indices,]
partial_x_train = x_train[-val_indices,]
y_val = y_train[val_indices,]
partial_y_train = y_train[-val_indices,]

#fit validation model
hist = model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val,y_val)
)

#############################################################################################
#model
model = keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

#compile
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
#model fit
model %>% fit(x_train,y_train,epochs=9, batch_size=512)
results = model %>% evaluate(x_test,y_test)
results
#prediction
prediction = model %>% predict(x_test)

#accuracy
py = prediction %>% apply(1,which.max)
py=py-1
sum(reuters$test$y==py)/2246
