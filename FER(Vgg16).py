#load the vgg model
vmodel = tf.keras.applications.vgg16.VGG16()


#see summary for vgg model
vmodel.summary()

#load the sequential model structure
model=Sequential()

#we are not using all of the layers from vgg original model, so we will remove some and add rest in our model
for i in vmodel.layers[:-1]:
    model.add(i)
model.summary()

#add the dense layer at last
model.add(Dense(7,activation='softmax'))

#compile the model
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#fit the model with training and validation set
model.fit(train_batches, 
                    steps_per_epoch=250, 
                    validation_data=valid_batches, 
                    validation_steps=85,
                    epochs=50,callbacks=[checkpoint])
