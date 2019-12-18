from keras.applications.xception import Xception
from keras.utils import plot_model
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import os
import numpy as np

#charge une image en 299*299 puis en forme une array de 
#pixel (rgb). On reduit les valeurs de RGB (0-255) à [-1, 1}
def process_img(path):
    img = image.load_img(path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    img -= 0.5
    img *= 2
    return img

#Formate les images contenues dans les paths via process_img
def process_imgs(paths):
    return np.concatenate([process_img(path) for path in paths])

#charge les images et en fait un vrai jeu de donnée
def load_data():
    #list les images contenues dans le dossier d
    def list_dir(d):
        xs = os.listdir(d)
        return [os.path.join(d, x) for x in xs]
    #Genere une matrice n * 2 ou n est 
    #le nombre d'element de x. chaque ligne est composé
    #d'un 0 ou d'un 1 : [[0,1], [1,0]...[1,0]]  
    def make_targets(x, y_class):
        y = np.zeros((x.shape[0], 5))
        y[:, y_class] = 1
        return y
    #creer un dataset composé des images et de leur resultat
    #exemple : tous les stégausores
    def make_dataset_part(d, y_class):
        x_train = process_imgs(list_dir(os.path.join(d, 'train')))
        x_test = process_imgs(list_dir(os.path.join(d, 'test')))
        return ((x_train, make_targets(x_train, y_class)),
                (x_test, make_targets(x_test, y_class)))
    #assemble les datasets 
    def glue_parts(parts):
        (x_train, y_train), (x_test, y_test) = parts[0]
        for (x_tr, y_tr), (x_te, y_te) in parts[1:]:
            x_train = np.concatenate([x_train, x_tr], axis=0)
            y_train = np.concatenate([y_train, y_tr], axis=0)
            x_test = np.concatenate([x_test, x_te], axis=0)
            y_test = np.concatenate([y_test, y_te], axis=0)
        return (x_train, y_train), (x_test, y_test)
    part_1 = make_dataset_part('pics/bulbasaur', 0)
    part_2 = make_dataset_part('pics/charmander', 1)
    part_3 = make_dataset_part('pics/mewtwo', 2)
    part_4 = make_dataset_part('pics/pikachu', 3)
    part_5 = make_dataset_part('pics/squirtle', 4)
    return glue_parts([part_1, part_2,  part_3,  part_4,  part_5 ])

#importe le modele Xception préentrainé sur imagenet en retirant la classification
base_model = Xception(weights='imagenet', include_top = False)
#recréé les deux couches de neurones manquanteset les ajoute au modèle
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='elu')(x)
predictions = Dense(5, activation='softmax')(x)
#Créé un objet modèle prenant les même input qu'Xception avec predictions pour base
model = Model(inputs=base_model.input, outputs=predictions)
#rend non-entrainable les couches natives
for layer in base_model.layers:
    layer.trainable = False
#decreasing learning rate
adam = Adam(lr=0.001)
#compilation du modèle
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = load_data()
#augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=10,
                    validation_data=(x_test, y_test), shuffle=True)
