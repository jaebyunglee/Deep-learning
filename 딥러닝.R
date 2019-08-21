install.packages('devtools')
devtools::install_github("rstudio/keras")
library(keras)
install_keras()


mnist <- dataset_mnist()

model = keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = c(784)) %>%
  layer_dense(units = 10, activation = "softmax")
