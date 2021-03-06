# bone_orbital_biomarker_network
Neural network to search for biomarkers limiting the bone orbit 

# Описание

Данная нейронная сеть позволяет определять координаты точек, ограничивающих костную глазницу человека. Результат её работы используется при моделировании модели объёмной глазницы и дальнейшей диагностики её повреждений.

## Технологии
Всё написано на python3 c использованием `Keras` :)

## Структура
Структура сети представляет из себя 16 слоёв:

```python
    Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 497, 497, 16)      12304     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 248, 248, 16)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 241, 241, 32)      32800     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 120, 120, 32)      0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 116, 116, 32)      25632     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 58, 58, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 57, 57, 64)        8256      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 26, 26, 128)       73856     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 13, 13, 128)       0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 11, 11, 256)       295168    
_________________________________________________________________
flatten_1 (Flatten)          (None, 30976)             0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 30976)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               15860224  
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 2056      
=================================================================
```

# Результат работы сети 
Резльтаты работы сети через приложение разметки(`VIA`):

<p align="center">    
<img src="https://github.com/birallex/bone_orbital_biomarker_network/blob/main/output/examples/example_1.png" width="512" height="512"/>
</p>

<p align="center">    
<img src="https://github.com/birallex/bone_orbital_biomarker_network/blob/main/output/examples/example_2.png" width="512" height="512"/>
</p>

<p align="center">    
<img src="https://github.com/birallex/bone_orbital_biomarker_network/blob/main/output/examples/example_3.png" width="512" height="512"/>
</p>