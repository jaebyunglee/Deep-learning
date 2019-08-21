rm(list=ls())
library(keras)
mnist = dataset_mnist()

train_images = mnist$train$x
train_labels = mnist$train$y
test_images = mnist$test$x
test_labels = mnist$test$y

#neural network architecture-----page 34
network = keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = "softmax")

#compile-----page 34
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#prepare image data-----page 34
train_images = array_reshape(train_images,c(60000,28*28)) #Convert a array to a matrix
train_images = train_images/255
test_images = array_reshape(test_images,c(10000,28*28))
test_images = test_images/255

#prepare lables-----page 35
train_labels = to_categorical(train_labels) #Convert a vector to a matrix
test_labels = to_categorical(test_labels)

#training-----page 35
network %>% fit(train_images, train_labels, epochs=5, batch_saze=128)

#test-----page 35
matrics = network %>% evaluate(test_images,test_labels)
matrics

#prediction-----page 36
network %>% predict_classes(test_images[1:10,])

#fifth sample of dataset
train_images[5,,]
digit = train_images[5,,]
plot(as.raster(digit,max=255))

#slice
my_slice = train_images[10:99,,]
my_slice = train_images[,15:28,15:28]

x = array(round(runif(1000,0,9)),dim=c(64,3,32,10))
