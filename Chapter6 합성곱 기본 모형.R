rm(list=ls())

library(keras)

#cnn & maxpooling
model = keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu",
                input_shape = c(28,28,1)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filter = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filter = 64, kernel_size = c(3,3), activation = "relu")

#dense
model = model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

model

mnist = dataset_mnist()
train_images = array_reshape(mnist$train$x, c(60000,28,28,1))
train_images = train_images/255
test_images = array_reshape(mnist$test$x, c(10000,28,28,1))
test_images = test_images/255

train_labels = to_categorical(mnist$train$y)
test_labels = to_categorical(mnist$test$y)

model %>% compile(
  optimizer = "Adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(train_images,train_labels,epochs = 5, batch_size=256)

pred = predict(model, test_images)
mean(mnist$test$y==(apply(pred,1,which.max)-1))
