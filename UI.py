# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(282, 351)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 20, 221, 291))
        self.groupBox.setObjectName("groupBox")
        self.btnFindCorners = QtWidgets.QPushButton(self.groupBox)
        self.btnFindCorners.setEnabled(True)
        self.btnFindCorners.setGeometry(QtCore.QRect(20, 30, 180, 25))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnFindCorners.sizePolicy().hasHeightForWidth())
        self.btnFindCorners.setSizePolicy(sizePolicy)
        self.btnFindCorners.setObjectName("btnFindCorners")
        self.btnFindIntrinsic = QtWidgets.QPushButton(self.groupBox)
        self.btnFindIntrinsic.setGeometry(QtCore.QRect(20, 70, 180, 25))
        self.btnFindIntrinsic.setObjectName("btnFindIntrinsic")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(15, 110, 190, 85))
        self.groupBox_2.setObjectName("groupBox_2")
        self.teSelectImage = QtWidgets.QTextEdit(self.groupBox_2)
        self.teSelectImage.setGeometry(QtCore.QRect(120, 15, 41, 30))
        self.teSelectImage.setObjectName("teSelectImage")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(30, 20, 85, 25))
        self.label.setObjectName("label")
        self.btnFindExtrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.btnFindExtrinsic.setGeometry(QtCore.QRect(30, 50, 150, 25))
        self.btnFindExtrinsic.setObjectName("btnFindExtrinsic")
        self.btnFindDistortion = QtWidgets.QPushButton(self.groupBox)
        self.btnFindDistortion.setGeometry(QtCore.QRect(20, 205, 180, 25))
        self.btnFindDistortion.setObjectName("btnFindDistortion")
        self.btnShowResult = QtWidgets.QPushButton(self.groupBox)
        self.btnShowResult.setGeometry(QtCore.QRect(20, 245, 180, 25))
        self.btnShowResult.setObjectName("btnShowResult")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 282, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2021 Opencvdl Hw2"))
        self.groupBox.setTitle(_translate("MainWindow", "2. Calibration"))
        self.btnFindCorners.setText(_translate("MainWindow", "2.1 Find Corners"))
        self.btnFindIntrinsic.setText(_translate("MainWindow", "2.2 Find Intrinsic"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.teSelectImage.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>"))
        self.label.setText(_translate("MainWindow", "Select image:"))
        self.btnFindExtrinsic.setText(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.btnFindDistortion.setText(_translate("MainWindow", "2.4 Find Distortion"))
        self.btnShowResult.setText(_translate("MainWindow", "2.5 Show Result"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
