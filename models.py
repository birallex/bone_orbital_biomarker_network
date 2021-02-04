from keras import layers, models, optimizers

def build_model_6():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (16, 16),activation = 'relu', input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (8, 8), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))   
    model.add(layers.Conv2D(64, (2, 2), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
    
    model.add(layers.Flatten())
    #-------------------------------------------------------------------
    model.add(layers.Dropout(0.5))
    #-------------------------------------------------------------------
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(8))

    print(model.summary())

    return model

def build_model_8():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (16, 16),activation = 'relu', input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (8, 8), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))   
    model.add(layers.Conv2D(64, (2, 2), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
    
    model.add(layers.Flatten())
    #-------------------------------------------------------------------
    model.add(layers.Dropout(0.5))
    #-------------------------------------------------------------------
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(8))

    print(model.summary())

    return model