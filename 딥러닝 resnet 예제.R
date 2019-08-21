library(keras)
library(ANTsRNet)


###ResNet


# instantiate the model
model <- application_resnet50(weights = 'imagenet')

# load the image
img_path <- "elephant.jpg"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)

# ensure we have a 4d tensor with single element in the batch dimension,
# the preprocess the input for prediction using resnet50
x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)

# make predictions then decode and print them
preds <- model %>% predict(x)
imagenet_decode_predictions(preds, top = 3)[[1]]
# }



###AlexNet

mnistData <- dataset_mnist()
numberOfLabels <- 10

# Extract a small subset for something that can run quickly

X_trainSmall <- mnistData$train$x[1:100,,]
X_trainSmall <- array( data = X_trainSmall, dim = c( dim( X_trainSmall ), 1 ) )
Y_trainSmall <- to_categorical( mnistData$train$y[1:100], numberOfLabels )

X_testSmall <- mnistData$test$x[1:10,,]
X_testSmall <- array( data = X_testSmall, dim = c( dim( X_testSmall ), 1 ) )
Y_testSmall <- to_categorical( mnistData$test$y[1:10], numberOfLabels )

# We add a dimension of 1 to specify the channel size

inputImageSize <- c( dim( X_trainSmall )[2:3], 1 )

model <- createAlexNetModel2D( inputImageSize = inputImageSize,
                               numberOfClassificationLabels = numberOfLabels )

model %>% compile( loss = 'categorical_crossentropy',
                   optimizer = optimizer_adam( lr = 0.0001 ),
                   metrics = c( 'categorical_crossentropy', 'accuracy' ) )

track <- model %>% fit( X_trainSmall, Y_trainSmall, verbose = 1,
                        epochs = 2, batch_size = 20, shuffle = TRUE, validation_split = 0.25 )

# Now test the model

testingMetrics <- model %>% evaluate( X_testSmall, Y_testSmall )
predictedData <- model %>% predict( X_testSmall, verbose = 1 )

