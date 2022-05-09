import numpy as np
import cv2, os, random, shutil, glob
from shutil import copyfile
from PyQt5 import QtWidgets
from UI5 import Ui_MainWindow
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.btnShowModelStructure.clicked.connect(self.btnShowModelStructureClicked)
        self.ui.btnShowTensorBoard.clicked.connect(self.btnShowTensorBoardClicked)
        self.ui.btnTest.clicked.connect(self.btnTestClicked)
        self.ui.btnDataAugmentation.clicked.connect(self.btnDataAugmentationClicked)
    # 5.1
    def btnShowModelStructureClicked(self):
        model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)
        print(model.summary())
    # 5.2
    def btnShowTensorBoardClicked(self):
        img=cv2.imread('./tensorboard.jpg')
        cv2.imshow('tensorboard', img)
    # 5.3
    def btnTestClicked(self):
        model=load_model('.\ResNet50.h5')
        category=random.randrange(2)
        if category==0:
            testImgs = glob.glob('.\\testing\dogs\*.jpg')
        else:
            testImgs = glob.glob('.\\testing\cats\*.jpg')
        img = image.load_img(testImgs[random.randrange(len(testImgs))], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        plt.imshow(img)
        feature=model.predict(x)
        if feature[0]>0.5:
            plt.title("dog")
        else:
            plt.title("cat")
        plt.axis('off')
        plt.show()
    # 5.4
    def btnDataAugmentationClicked(self):
        img=cv2.imread('./table of accuracy.png')
        cv2.imshow('table of accuracy', img)
    
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())