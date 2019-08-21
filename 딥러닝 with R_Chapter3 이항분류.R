rm(list=ls())
library(keras)

#data
imdb = dataset_imdb(num_words = 10000)

train_data = imdb$train$x
train_labels = imdb$train$y
test_data = imdb$test$x
test_labels = imdb$test$y

str(train_data[[1]])
train_labels[1]

#vectorization
vec_seq.fun = function(data,dim=10000){
  result = matrix(0,length(data),10000)
  for(i in 1:length(data)){
    result[i,data[[i]]] = 1
  }
  return(result)
}



x_train = vec_seq.fun(train_data,dim=10000)
memory.size()
memory.limit(5000)
x_test = vec_seq.fun(test_data,dim=10000)
str(x_train[1,]); str(x_test[1,])
y_train = as.numeric(train_labels)
y_test = as.numeric(test_labels)

#neural network
model = keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

#compile
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#validation
val_indices = 1:10000
x_val = x_train[val_indices,]
partial_x_train = x_train[-val_indices,]
y_val = y_train[val_indices]
partial_y_train = y_train[-val_indices]

#training model
history = model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val,y_val)
)

plot(history)

# model
model = keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(x_train,y_train,epochs = 4, batch_size = 512)
results = model %>% evaluate(x_test,y_test)
results


