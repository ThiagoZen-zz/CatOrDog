
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_gato_cachorro.json','r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_gato_cachorro.h5')

#GATO --> dataset/test_set/gato/cat.3500.jpg
#CACHORRO --> dataset/test_set/cachorro/dog.3500.jpg

imagem_teste = image.load_img('dataset/test_set/gato/cat.3963.jpg',target_size = (64,64))

imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste,axis = 0)

previsao = classificador.predict(imagem_teste)
print(previsao)
print('é um gato? :' + str(previsao > 0.5) )