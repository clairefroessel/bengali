import pandas as pd
from model import ResNet50
from data_augmentation import MyDataGenerator
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
EPOCHS = 15
SIZE = 224


train_files, valid_files = train_test_split(train, test_size=0.25, 
    random_state=2019)


train_datagen = MyDataGenerator(cutmix_alpha = 0.4, zoom_range = [0.5,1], rotation_range = 90)
valid_datagen = MyDataGenerator(cutmix_alpha = 0, zoom_range = 0, rotation_range = 0)

train_generator = train_datagen.myflow(train_files, batch_size = BATCH_SIZE)
valid_generator = valid_datagen.myflow(valid_files, batch_size = BATCH_SIZE)

train_steps = round(len(train_files) / BATCH_SIZE) + 1
valid_steps = round(len(valid_files) / BATCH_SIZE) + 1


model= ResNet50(input_shape = (SIZE, SIZE, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  

train_history = model.fit_generator(train_generator, 
                                    steps_per_epoch=train_steps, 
                                    epochs=EPOCHS, 
                                    validation_data=valid_generator,
                                    validation_steps = valid_steps,verbose = 1)