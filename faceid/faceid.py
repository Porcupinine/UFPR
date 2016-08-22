#!/usr/bin/python

# Import the required modules
import cv2, os
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import numpy.linalg as npla

class FaceId:
    _path = ''  ## virutal path
    _exclude = ''
    _accuracy_list = []

    def __init__(self, path=_path, exclude=_exclude, accuracy_list= _accuracy_list):
        self.path = path
        self.exclude = exclude
        self.accuracy_list = accuracy_list
        self.get_images_and_labels(self.path, exclude)

    def eigenFaces2(self):
        # 		cria uma variavel com a quantidade de imagens
        nimg = len(self.images_train)
        nimg_test = len(self.images_test)
        #		cria duas variaveis com o tamanho das imagens
        n1, n2 = self.images_train[0].shape
        #		Arreio com as imagens
        aimages = np.asarray(self.images_train)
        aimages_test = np.asarray(self.images_test)
        #		transforma em um arreio com uma dimensao
        aimages = np.reshape(np.ravel(aimages), (nimg, -1))
        aimages_test = np.reshape(np.ravel(aimages_test), (nimg_test, -1))
        #		calculaa media das imagens
        mimages = np.average(aimages, axis=0)
        #		arreio com a diferenca entre imagem e imagem media (ri nos slides)
        dimages = aimages - mimages
        dimages_test = aimages_test - mimages
        #		matriz de covariancia
        mcov = np.cov(dimages)
        #		auto-valor e auto-vetor da matriz de covariancia
        evals, evects = npla.eig(mcov)
        evects = np.real(evects)

        #		Cria lista de eigenfaces
        eigenFaces = []
        ims = None  # for exhibition
        #		realiza a operacao para 5 imagens
        for i in range(5):
            #			lista recebe a multiplicacao dos auto-vetores pela diferenca entre imagens e imagens medias
            eigenFace = np.dot(evects[i, :], dimages)
            #			troca o tamanho do arreio
            eigenFace = np.array(eigenFace, dtype=np.uint8).reshape(n1, n2)
            #			acrescenta na lista
            eigenFaces.append(eigenFace)
        # Exibe imagem resultante
        # if ims is None:
        #	ims = plt.imshow(eigenFace) #, cmap='Greys_r')
        # else:
        #	ims.set_data(eigenFace)
        # plt.pause(2)
        # plt.draw()

        aeigenFaces = np.asarray(eigenFaces)
        nimg_eg = len(aeigenFaces)
        aeigenFaces = np.reshape(np.ravel(aeigenFaces), (nimg_eg, -1))
        #aeigenFaces = aeigenFaces.T
        dimages = dimages.T
        dimages_test = dimages_test.T
        #aimages_test = aimages_test.T

        #distancias base de treino e test
        pvec_train = np.dot(aeigenFaces, dimages)
        pvec_test = np.dot(aeigenFaces, dimages_test)


        threshold = 0
        for i in pvec_train.T:
            for j in pvec_train.T:
                threshold = max (threshold, np.sqrt(np.sum(np.square(i-j)))/2 )
        #print threshold

        #classificacao
        distvec = []
        classified = []
        for i in pvec_test.T:
            for j in pvec_train.T:
                distvec.append(np.sqrt(np.sum(np.square(i-j))))
            classe = (self.subjects_train[distvec.index(min(distvec))])
            distvec[:] = []
            classified.append(classe)

        accuracy = 0
        for i in range(len(classified)):
            if classified[i] == self.subjects_test[i]:
                accuracy = accuracy +1
        accuracy = accuracy*100/len(classified)
        self.accuracy_list.append(accuracy)
        print "Accuracy %d%% \n" %accuracy
        #		retorna lista com eigenfaces
        return eigenFaces


class ORLFaces(FaceId):
    _path = './att_faces'
    _exclude = ''

    #	funcao para capturar as imagens e "etiquetas" por que "path = _path" ??
    def get_images_and_labels(self, path=_path, exclude=_exclude):

        self.images_train = []
        self.subjects_train = []
        self.images_test = []
        self.subjects_test = []

        #		percorre as pastas
        subject_paths = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for d in subject_paths:
        #print subject_path
            for f in os.listdir(d):
                if f.endswith('.pgm') and os.path.isfile(os.path.join(d, f)):
                    if f.split(".")[0] == exclude:
                        image_path = os.path.join(d, f)
                        # Read the image and convert to grayscale
                        image_pil = Image.open(image_path).convert('L')
                        # Convert the image format into numpy array
                        image = np.array(image_pil, 'uint8')
                        # Get the label of the image
                        nbr = int(os.path.split(d)[1].replace("s", ""))

                        self.images_test.append(image)
                        self.subjects_test.append(nbr)
                    else:
                        image_path = os.path.join(d, f)
                        # Read the image and convert to grayscale
                        image_pil = Image.open(image_path).convert('L')
                        # Convert the image format into numpy array
                        image = np.array(image_pil, 'uint8')
                        # Get the label of the image
                        nbr = int(os.path.split(d)[1].replace("s", ""))

                        self.images_train.append(image)
                        self.subjects_train.append(nbr)

class YaleFaces(FaceId):
    _path = './yalefaces'
    _exclude = ''

    def get_images_and_labels(self, path=_path, exclude=_exclude):
        # images_train will contains face images for training
        self.images_train = []
        # subjets_train will contains the subject identification number assigned to the image for training
        self.subjects_train = []
        # classes_train for training
        self.classes_train = []
        # images_test will contains face images for testing
        self.images_test = []
        # subjects_test will contains the subjects identifications number assigned to the image for testing
        self.subjects_test = []
        # classes_test for testing
        self.classes_test = []


        for f in os.listdir(path):

            if f.endswith(exclude):
                image_paths = os.path.join(path, f)
                # Read the image and convert to grayscale
                image_pil = Image.open(image_paths).convert('L')
                # Convert the image format into numpy array
                image = np.array(image_pil, 'uint8')
                # Get the label of the image
                nbr = int(os.path.split(image_paths)[1].split(".")[0].replace("subject", ""))

                self.images_test.append(image)
                self.subjects_test.append(nbr)
                self.classes_test.append(exclude)


            elif os.path.isfile(os.path.join(path, f)):
                image_paths = os.path.join(path, f)
                image_pil = Image.open(image_paths).convert('L')
                image = np.array(image_pil, 'uint8')
                nbr = int(os.path.split(image_paths)[1].split(".")[0].replace("subject", ""))
                class_label = '.' +  f.split(".")[1]

                self.images_train.append(image)
                self.subjects_train.append(nbr)
                self.classes_train.append(class_label)

path = './att_faces'
print 'For ORL base \n'
class_labels = ['1','2','3','4','5','6','7','8','9','10']
ORL_accuracy_list =[]
for y in class_labels:
    print 'image %s' %y
    orl = ORLFaces(path, y, ORL_accuracy_list)
    orl.eigenFaces2()
best = (class_labels[ORL_accuracy_list.index(max(ORL_accuracy_list))])
best_value = (max(ORL_accuracy_list))
worst = (class_labels[ORL_accuracy_list.index(min(ORL_accuracy_list))])
worst_value = (min(ORL_accuracy_list))
print 'best result for image %s, accuracy = %d%% \n' % (best, best_value)
print 'worst result for image %s, accuracy = %d%% \n' % (worst, worst_value)

path = './yalefaces'
print 'For Yale  base \n'
class_labels = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses', '.normal', '.rightlight', '.sad',
                '.sleepy', '.surprised', '.wink']
Yale_accuracy_list = []
for x in class_labels:
    print 'subclasse %s' %x
    yale = YaleFaces(path, x, Yale_accuracy_list)
    yale.eigenFaces2()
best = (class_labels[Yale_accuracy_list.index(max(Yale_accuracy_list))])
best_value = (max(Yale_accuracy_list))
worst = (class_labels[Yale_accuracy_list.index(min(Yale_accuracy_list))])
worst_value = (min(Yale_accuracy_list))
print 'best result for image %s, accuracy = %d%% \n' % (best, best_value)
print 'worst result for image %s, accuracy = %d%% \n' % (worst, worst_value)













