rm(list=ls())
library(keras)


base_dir <- "C:/Users/kis91/Desktop/R공부/딥러닝/kaggle"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")
datagen <- image_data_generator(rescale = 1/255)
batch_size <- 32

conv_base = application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224,224,3)
)



extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 7, 7, 512))  
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(224, 224),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch) #conv_base
    
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      # Note that because generators yield data indefinitely in a loop, 
      # you must break after every image has been seen once.
      break
  }
  
  list(
    features = features, 
    labels = labels
  )
}


train <- extract_features(train_dir, 2002) #number of samples
validation <- extract_features(validation_dir, 600)
test <- extract_features(test_dir, 400)

#reshape features
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 7 * 7 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)




####vgg16 + FCL2######
model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu",  #FCL1
              input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") #FCL2
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#find opt epochs
history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)