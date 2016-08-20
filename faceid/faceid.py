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

    def __init__(self, path=_path, exclude=_exclude):
        self.path = path
        self.exclude = exclude
        self.get_images_and_labels(self.path, exclude)

    def meanFace(self):
        #		print '{0}'.format(len(self.images))
        #		cria um arreio com o tamanho das imagens
        imgmf = np.zeros(self.images[0].shape, dtype=np.uint32)  # due to integer summations uint32
        #		soma todas as imagens ao arreio	(soma as matrizes)
        for im in self.images:
            imgmf = imgmf + im
        # divide a soma das matrizes pela quantidade de imagens utilizando o tamanho da lista images
        imgmf = imgmf / len(self.images)
        imgmf = np.array(imgmf, dtype=np.uint8)  # converting back to uin8
        #		retorna imagem media
        return imgmf

    def meanFace2(self):
        #		cria um arreio com as imagens ???
        aimages = np.array(self.images)
        #		imgmf = np.mean(aimages,axis=0,dtype=np.uint8) # computes the average face
        #		calcula a imagem media
        imgmf = np.average(aimages, axis=0)  # computes the average face
        imgmf = np.array(imgmf, dtype=np.uint8)
        #		retorna a imagem media
        return imgmf

    def eigenFaces(self):
        # covariance matrix
        nimg = len(self.images)
        aimages = np.asarray(self.images)
        aimages = np.reshape(np.ravel(aimages), (nimg, -1))

    # def eigenFaces2(self):
    #	ret_train = eigenFacesDo(self.images_train)
    #	ret_test = eigenFacesDo(self.images_test)

    def eigenFaces2(self):
        # 		cria uma variavel com a quantidade de imagens
        nimg = len(self.images_train)
        nimg_test = len(self.images_test)
        #		cria duas variaveis com o tamanho das imagens ??
        n1, n2 = self.images_train[0].shape
        #		Arreio com as imagens
        aimages = np.asarray(self.images_train)
        aimages_test = np.asarray(self.images_test)
        #print 'aimages_test shape: {0}'.format(aimages_test.shape)
        #		transforma em um arreio com uma dimensao??? (o que faz ravel ??)
        aimages = np.reshape(np.ravel(aimages), (nimg, -1))
        aimages_test = np.reshape(np.ravel(aimages_test), (nimg_test, -1))
        #print 'aimages_test shape: {0}'.format(aimages_test.shape)
        #		calculaa media das imagens
        mimages = np.average(aimages, axis=0)
        #		arreio com a diferenca entre imagem e imagem media (ri nos slides)
        dimages = aimages - mimages
        dimages_test = aimages_test - mimages
        # print dimages_test
        #print 'shape: {0}/{1}'.format(dimages_test.shape, dimages_test.dtype)
        #		matriz de covariancia
        mcov = np.cov(dimages)
        #		auto-valor e auto-vetor da matriz de covariancia
        evals, evects = npla.eig(mcov)
        evects = np.real(evects)
      #  evects = np.sort(evects)
        # print 'shape: {0}'.format(mcov.shape)
        # print 'shape: {0}'.format(evals.shape)
        # print 'shape: {0}'.format(evects.shape)
        #		Cria lista de eigenfaces
        eigenFaces = []
        ims = None  # for exhibition
        #		realiza a operacao para 5 imagens
        for i in range(5):
            #print 'shape: {0}'.format(aimages.shape)
            #print 'shape: {0}/{1}'.format(evects.shape, evects.dtype)
            #			lista recebe a multiplicacao dos auto-vetores pela diferenca entre imagens e imagens medias
            eigenFace = np.dot(evects[i, :], dimages)
            #print 'shape: {0}/{1}'.format(eigenFace.shape,eigenFace.dtype)
            #			troca o tamanho do arreio ???
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

        print 'Eigen Faces pronto'
        #dimages_test = np.array(dimages_test, dtype=np.uint8).reshape(n1, n2)

        aeigenFaces = np.asarray(eigenFaces)
        nimg_eg = len(aeigenFaces)
        aeigenFaces = np.reshape(np.ravel(aeigenFaces), (nimg_eg, -1))
        #aeigenFaces = aeigenFaces.T
        dimages = dimages.T
        dimages_test = dimages_test.T
        #aimages_test = aimages_test.T

        #print 'shape: {0}'.format(aeigenFaces.shape)
        #print 'shape: {0}'.format(dimages_test.shape)

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
                #dist = np.sqrt(np.sum(np.square(i-j)))
                distvec.append(np.sqrt(np.sum(np.square(i-j))))
                #dist = min (dist, np.sqrt(np.sum(np.square(i-j))))
                #classified.append()
            classe = (self.subjects_train[distvec.index(min(distvec))])
            distvec[:] = []
            classified.append(classe)
        #classified.append(self.subjects_train.index(min(distvec)))

        acuracy = 0
        for i in range(len(classified)):
            if classified[i] == self.subjects_test[i]:
                acuracy = acuracy +1
        print acuracy
        #print classified
        #		retorna lista com eigenfaces
        return eigenFaces
        # def fisherFaces(self):


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
        subjects_paths = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        #		enumera as imagens da pasta (cada pasta tem um numero) o que eh o s ????
        for s, subject_paths in enumerate(subjects_paths, start=1):
            subject_path = [os.path.join(subject_paths, f) for f in os.listdir(subject_paths) if
                            f.endswith('.pgm') and os.path.isfile(os.path.join(subject_paths, f))]
            if f.startswith(exclude):

                for image_path in subject_path:
                    # print 'sub: {0}'.format(image_path)
                    # Read the image and convert to grayscale
                    image_pil = Image.open(image_path).convert('L')
                    # Convert the image format into numpy array
                    image = np.array(image_pil, 'uint8')
                    # Get the label of the image
                    nbr = int(os.path.split(subject_path)[1].split(".")[0])

                    self.images_test.append(image)
                    self.subjects_test.append(nbr)
                    #print 'test done'
            else:

                for image_path in subject_path:
                    # print 'sub: {0}'.format(image_path)
                    # Read the image and convert to grayscale
                    image_pil = Image.open(image_path).convert('L')
                    # Convert the image format into numpy array
                    image = np.array(image_pil, 'uint8')
                    # Get the label of the image
                    nbr = int(os.path.split(subject_path)[1].split(".")[0])

                    self.images_train.append(image)
                    self.subjects_train.append(nbr)
                    # print 'train done'
                    # print self.images_train[0]


# print 'sub: {0}({1}#) - {2}'.format(s,len(subject_path),subject_paths)


class YaleFaces(FaceId):
    _path = './yalefaces'
    _exclude = ''
    # classes: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.
    class_labels = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses', '.normal', '.rightlight', '.sad',
                    '.sleepy', '.surprised', '.wink']

    # Note that the image "subject04.sad" has been corrupted and has been substituted by "subject04.normal".
    # Note that the image "subject01.gif" corresponds to "subject01.centerlight" :~ mv subject01.gif subject01.centerlight


    def get_images_and_labels(self, path=_path, exclude=_exclude):
        # Append all the absolute image paths in a list image_paths
        # We will not read the image with the .sad extension in the training set
        # Rather, we will use them to test our accuracy of the training
        #print 'entrou'
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

        #for c, class_label in enumerate(self.class_labels, start=1):
            # image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(class_label)]

            #			separar arquivo com imagens selecionadas e arquivo com demais imagens
            # print c
            # print class_label

        for f in os.listdir(path):
            # print f

            if f.endswith(exclude):
                # print 'endsWith something'
                image_paths = os.path.join(path, f)
                #					print 'Image: ' + image_path
                # Read the image and convert to grayscale
                image_pil = Image.open(image_paths).convert('L')
                # Convert the image format into numpy array
                image = np.array(image_pil, 'uint8')
                # Get the label of the image
                nbr = int(os.path.split(image_paths)[1].split(".")[0].replace("subject", ""))

                self.images_test.append(image)
                self.subjects_test.append(nbr)
                self.classes_test.append(exclude)

            # print 'test done'
            # print len(self.images_test)

            elif os.path.isfile(os.path.join(path, f)):
                image_paths = os.path.join(path, f)
                # print 'else'
                # print image_paths
                # print f
                image_pil = Image.open(image_paths).convert('L')
                image = np.array(image_pil, 'uint8')
                nbr = int(os.path.split(image_paths)[1].split(".")[0].replace("subject", ""))
                class_label = '.' +  f.split(".")[1]

                self.images_train.append(image)
                self.subjects_train.append(nbr)
                self.classes_train.append(class_label)

        #	print 'class_label: {0}({1}#) - {2}'.format(c,len(image_paths), class_label)


## Path to the Yale Dataset
# path = '/home/menotti/databases/yalefaces/'
# print 'loading Yalefaces database'
# yale = YaleFaces(path)

# path = '/home/menotti/databases/orl_faces/'
# print 'loading ORL database'
# orl = ORL(path)

#		ims = None # for exhibition
#			if ims is None:
#				ims = plt.imshow(im, cmap='Greys_r')
#			else:
#				ims.set_data(im)
#			plt.pause(.01)
#			plt.draw()


# path = '/home/prof/menotti/databases/yalefaces/'
# print 'loading Yalefaces database'
# yale = YaleFaces(path)
# yale.eigenFaces2()

path = './att_faces'
print 'loading ORL database'
# class_labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40']
# for y in range(len(class_labels)):
#	orl = ORLFaces(path, y)
#	orl.eigenFaces2()

path = './yalefaces'
class_labels = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses', '.normal', '.rightlight', '.sad',
                '.sleepy', '.surprised', '.wink']
for x in class_labels:
    yale = YaleFaces(path, x)
    yale.eigenFaces2()












