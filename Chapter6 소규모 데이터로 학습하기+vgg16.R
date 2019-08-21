rm(list=ls())
library(keras)

datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# !!!!!! data load
fnames = list.files("C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/kaggle/train/cat", full.names = TRUE)
img = image_load(fnames, target_size = c(224,224))
img_array = image_to_array(img)
img_array = array_reshape(img_array,c(1,224,224,3))

#data generator
augmentation_generator = flow_images_from_data(
  img_array,
  generator = datagen,
  batch_size = 1
)

#plot cat
op = par(mfrow = c(3,2), pty="s",mar=c(1,0,1,0))
for(i in 1:6){
  batch = generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}

###############################vgg16############################################

###1 augmenting
datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen = image_data_generator(rescale = 1/255)




###2
train_generator = flow_images_from_directory(
  "C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/kaggle/train",
  datagen,
  target_size = c(224,224),
  batch_size = 32,
  class_mode = "binary"
)

valid_generator = flow_images_from_directory(
  "C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/kaggle/valid",
  test_datagen,
  target_size = c(224,224),
  batch_size = 32,
  class_mode = "binary"
)

test_generator = flow_images_from_directory(
  "C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/kaggle/test",
  test_datagen,
  target_size = c(224,224),
  batch_size = 32,
  class_mode = "binary"
)





###3 vgg16
conv_base = application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224,224,3)
)

model = keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")





#freezing, before compile
freeze_weights(conv_base)
model$trainable_weights #trainable weights

#compile
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0002),
  metric = c("accuracy")
)


#fit
history = model %>% fit_generator(
  train_generator,
  steps_per_epoch = 63,
  epochs = 10,
  validation_data = valid_generator,
  validation_steps = 50
)

#optimal epochs
opt.epochs = which.max(history$metrics$val_acc)

#model save
#model %>% save_model_hdf5("C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/cats_and_dogs_1.h5")
load.model=load_model_hdf5("C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/cats_and_dogs_1.h5")

evaluate_generator(load.model,test_generator,steps = 13) #acc 0.8843374



##############################################################fine-tuning
### unfreezing
unfreeze_weights(conv_base, from = "block3_conv1")
model$trainable_weights

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0002),
  metric = c("accuracy")
)


#fit
history = model %>% fit_generator(
  train_generator,
  steps_per_epoch = 63,
  epochs = 10,
  validation_data = valid_generator,
  validation_steps = 50
)


model %>% save_model_hdf5("C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/cats_and_dogs_2.h5")
load.model=load_model_hdf5("C:/Users/kis91/Desktop/R°øºÎ/µö·¯´×/cats_and_dogs_2.h5")

evaluate_generator(load.model,test_generator,steps = 13) #acc 0.8843374