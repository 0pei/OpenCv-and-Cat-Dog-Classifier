import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PyQt5 import QtWidgets
from scipy import signal
from UI import Ui_MainWindow
from PIL import Image

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.btnFindCorners.clicked.connect(self.btnFindCornersClicked)
        self.ui.btnFindIntrinsic.clicked.connect(self.btnFindIntrinsicClicked)
        self.ui.btnFindExtrinsic.clicked.connect(self.btnFindExtrinsicClicked)
        self.ui.btnFindDistortion.clicked.connect(self.btnFindDistortionClicked)
        self.ui.btnShowResult.clicked.connect(self.btnShowResultClicked)
    # 2.1
    def btnFindCornersClicked(self):
        # for no in range(1,16):
        #     path=".\Q2_Image\\"+str(no)+".bmp"
        #     img=cv2.imread(path)
        #     retval,corners=cv2.findChessboardCorners(img,(11,8),flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        #     cv2.drawChessboardCorners(img,(11,8),corners,retval)
        #     img=cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
        #     cv2.imshow('bmp', img)
        #     cv2.waitKey(500)

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.images = glob.glob('.\Q2_Image\*.bmp')
        for fname in self.images:
            img = cv2.imread(fname)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(self.gray, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(self.gray, corners, (11,8), (-1,-1), criteria)
                self.imgpoints.append(corners)
                cv2.drawChessboardCorners(img, (11,8), corners2, ret)
                # cv2.drawChessboardCorners(img, (11,8), corners, ret)
                img=cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    # 2.2
    def btnFindIntrinsicClicked(self):
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        print("Intrinsic:")
        print(self.mtx)
    # 2.3
    def btnFindExtrinsicClicked(self):
        imageNo=int(self.ui.teSelectImage.toPlainText())
        R=cv2.Rodrigues(self.rvecs[imageNo-1])[0]
        t=self.tvecs[imageNo-1]
        Rt=np.concatenate([R,t], axis=-1)   # [R|t]
        H=np.matmul(self.mtx,Rt)            # A[R|t]
        print("Extrinsic:")
        print(Rt)
        # Rvecs = np.zeros((3, 3))
        # cv2.Rodrigues(self.rvecs[imageNo-1], Rvecs, jacobian=0)
        # extrinsic = np.hstack([Rvecs, self.tvecs[imageNo-1]])
        # print(extrinsic)
    # 2.4
    def btnFindDistortionClicked(self):
        print("Distortion:")
        print(self.dist)
    # 2.5  
    def btnShowResultClicked(self):
        for fname in self.images:
            img = cv2.imread(fname)
            h, w=img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 0, (w,h))
            undist=cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            x,y,w,h=roi
            undist=undist[y:y+h, x:x+w]
            img=cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
            undist=cv2.resize(undist, (800, 800), interpolation=cv2.INTER_AREA)
            imgs=np.hstack([img,undist])
            cv2.imshow("result", imgs)
            cv2.waitKey(500)
    
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())