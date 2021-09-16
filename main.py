from fastapi import FastAPI, File, UploadFile, Form
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Se almacena el modelo de ResNet50 a la variable "model".
model = ResNet50(weights="imagenet")

# Se inicia FastAPI
app = FastAPI()

# Definimos una funcion para conver una lista de tuplas en diccionario.
def Convert(tup, di):
    di = dict(tup)
    return di

@app.post("/file")
async def _file_upload(
    #Se recibe una imagen.
    my_file: UploadFile = File(...),
):
    # Se inserta la imagen en la variable img_path.
    img_path = my_file.file

    # conversion a formato especifico de imagen para su correcto funcionamiento.
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # resultado del modelo
    preds = model.predict(x)

    # Se obtienen las 3 primeras prediccionas para la imagen despues transformamos la lista en un diccionario.
    lst = decode_predictions(preds, top=3)[0]
    a_list = [a_tuple[1:] for a_tuple in lst]
    dictionary = {}
    dictionary = Convert(a_list, dictionary)

    for k, v in dictionary.items():
        dictionary[k] = float(v)

    # Se asigna el nombre de la imagen al diccionario.
    dictionary["name"] = my_file.filename

    #  Se retorna el diccionario con las predicciones y el nombre de la imagen antes dada.
    return dictionary