# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_spam.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_NaiveBayes_Spam_filter(object):
    def setupUi(self, NaiveBayes_Spam_filter):
        NaiveBayes_Spam_filter.setObjectName("NaiveBayes_Spam_filter")
        NaiveBayes_Spam_filter.resize(1200, 800)
        NaiveBayes_Spam_filter.setMinimumSize(QtCore.QSize(1200, 800))
        NaiveBayes_Spam_filter.setMaximumSize(QtCore.QSize(1200, 800))
        self.mainWidget = QtWidgets.QWidget(NaiveBayes_Spam_filter)
        self.mainWidget.setMinimumSize(QtCore.QSize(1200, 800))
        self.mainWidget.setMaximumSize(QtCore.QSize(1200, 800))
        self.mainWidget.setObjectName("mainWidget")
        self.label_1 = QtWidgets.QLabel(self.mainWidget)
        self.label_1.setGeometry(QtCore.QRect(0, 25, 1200, 50))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_1.setFont(font)
        self.label_1.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(self.mainWidget)
        self.label_2.setGeometry(QtCore.QRect(200, 750, 400, 50))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.mainWidget)
        self.label_3.setGeometry(QtCore.QRect(600, 750, 400, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.line_1 = QtWidgets.QFrame(self.mainWidget)
        self.line_1.setGeometry(QtCore.QRect(0, 740, 1200, 10))
        self.line_1.setStyleSheet("background-color: rgb(255, 255, 220);\n"
"color: rgb(0, 0, 255);")
        self.line_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_1.setLineWidth(2)
        self.line_1.setMidLineWidth(2)
        self.line_1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_1.setObjectName("line_1")
        self.exitButton = QtWidgets.QPushButton(self.mainWidget)
        self.exitButton.setGeometry(QtCore.QRect(825, 650, 150, 50))
        font = QtGui.QFont()
        font.setFamily("方正姚体")
        font.setPointSize(14)
        self.exitButton.setFont(font)
        self.exitButton.setStyleSheet("background-color: rgb(224, 222, 215);")
        self.exitButton.setObjectName("exitButton")
        self.line_2 = QtWidgets.QFrame(self.mainWidget)
        self.line_2.setGeometry(QtCore.QRect(0, 80, 1200, 10))
        self.line_2.setStyleSheet("background-color: rgb(255, 255, 220);\n"
"color: rgb(0, 0, 255);")
        self.line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_2.setLineWidth(2)
        self.line_2.setMidLineWidth(2)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setObjectName("line_2")
        self.trainBox = QtWidgets.QGroupBox(self.mainWidget)
        self.trainBox.setGeometry(QtCore.QRect(25, 125, 550, 600))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.trainBox.setFont(font)
        self.trainBox.setObjectName("trainBox")
        self.infoBrowser = QtWidgets.QTextBrowser(self.trainBox)
        self.infoBrowser.setGeometry(QtCore.QRect(75, 100, 400, 400))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.infoBrowser.setFont(font)
        self.infoBrowser.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(217, 255, 246, 255), stop:1 rgba(255, 255, 255, 255));")
        self.infoBrowser.setObjectName("infoBrowser")
        self.trainButton = QtWidgets.QPushButton(self.trainBox)
        self.trainButton.setGeometry(QtCore.QRect(100, 525, 150, 50))
        font = QtGui.QFont()
        font.setFamily("方正姚体")
        font.setPointSize(14)
        self.trainButton.setFont(font)
        self.trainButton.setStyleSheet("background-color: rgb(224, 222, 215);")
        self.trainButton.setObjectName("trainButton")
        self.label_4 = QtWidgets.QLabel(self.trainBox)
        self.label_4.setGeometry(QtCore.QRect(75, 25, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.trainBox)
        self.label_5.setGeometry(QtCore.QRect(75, 50, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_5.setObjectName("label_5")
        self.resettrainButton = QtWidgets.QPushButton(self.trainBox)
        self.resettrainButton.setGeometry(QtCore.QRect(300, 525, 150, 50))
        font = QtGui.QFont()
        font.setFamily("方正姚体")
        font.setPointSize(14)
        self.resettrainButton.setFont(font)
        self.resettrainButton.setStyleSheet("background-color: rgb(224, 222, 215);")
        self.resettrainButton.setObjectName("resettrainButton")
        self.label_11 = QtWidgets.QLabel(self.trainBox)
        self.label_11.setGeometry(QtCore.QRect(75, 75, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_11.setObjectName("label_11")
        self.groupBox = QtWidgets.QGroupBox(self.mainWidget)
        self.groupBox.setGeometry(QtCore.QRect(625, 125, 550, 500))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(0, 100, 550, 40))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(252, 255, 217, 255), stop:1 rgba(255, 255, 255, 255));")
        self.lineEdit.setObjectName("lineEdit")
        self.resultBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.resultBrowser.setGeometry(QtCore.QRect(75, 300, 400, 50))
        self.resultBrowser.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 217, 217, 255), stop:1 rgba(255, 255, 255, 255));")
        self.resultBrowser.setObjectName("resultBrowser")
        self.predictButton = QtWidgets.QPushButton(self.groupBox)
        self.predictButton.setGeometry(QtCore.QRect(100, 400, 150, 50))
        font = QtGui.QFont()
        font.setFamily("方正姚体")
        font.setPointSize(14)
        self.predictButton.setFont(font)
        self.predictButton.setStyleSheet("background-color: rgb(224, 222, 215);")
        self.predictButton.setObjectName("predictButton")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(0, 40, 550, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(0, 65, 550, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(75, 200, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(75, 225, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(75, 250, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_10.setObjectName("label_10")
        self.resetpredictButton = QtWidgets.QPushButton(self.groupBox)
        self.resetpredictButton.setGeometry(QtCore.QRect(300, 400, 150, 50))
        font = QtGui.QFont()
        font.setFamily("方正姚体")
        font.setPointSize(14)
        self.resetpredictButton.setFont(font)
        self.resetpredictButton.setStyleSheet("background-color: rgb(224, 222, 215);")
        self.resetpredictButton.setObjectName("resetpredictButton")
        self.resultBrowser.raise_()
        self.lineEdit.raise_()
        self.predictButton.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.resetpredictButton.raise_()
        NaiveBayes_Spam_filter.setCentralWidget(self.mainWidget)

        self.retranslateUi(NaiveBayes_Spam_filter)
        QtCore.QMetaObject.connectSlotsByName(NaiveBayes_Spam_filter)

    def retranslateUi(self, NaiveBayes_Spam_filter):
        _translate = QtCore.QCoreApplication.translate
        NaiveBayes_Spam_filter.setWindowTitle(_translate("NaiveBayes_Spam_filter", "MainWindow"))
        self.label_1.setText(_translate("NaiveBayes_Spam_filter", "项目名称：基于朴素贝叶斯分类算法的简易垃圾短信分类系统"))
        self.label_2.setText(_translate("NaiveBayes_Spam_filter", "作者：张智淋 2054169"))
        self.label_3.setText(_translate("NaiveBayes_Spam_filter", "by PyQt5.15.6、Qt Designer"))
        self.exitButton.setText(_translate("NaiveBayes_Spam_filter", "退出程序"))
        self.trainBox.setTitle(_translate("NaiveBayes_Spam_filter", "训练"))
        self.trainButton.setText(_translate("NaiveBayes_Spam_filter", "开始训练"))
        self.label_4.setText(_translate("NaiveBayes_Spam_filter", "点击“开始训练”按钮基于数据集进行训练，"))
        self.label_5.setText(_translate("NaiveBayes_Spam_filter", "训练相关提示信息将在下方文本框中显示。"))
        self.resettrainButton.setText(_translate("NaiveBayes_Spam_filter", "重置"))
        self.label_11.setText(_translate("NaiveBayes_Spam_filter", "（训练过程约2至5分钟）"))
        self.groupBox.setTitle(_translate("NaiveBayes_Spam_filter", "预测"))
        self.predictButton.setText(_translate("NaiveBayes_Spam_filter", "预测分类"))
        self.label_6.setText(_translate("NaiveBayes_Spam_filter", "以下黄色输入框输入一条待预测类别的英文短信"))
        self.label_7.setText(_translate("NaiveBayes_Spam_filter", "示例：Double your mins & txts on Orange"))
        self.label_8.setText(_translate("NaiveBayes_Spam_filter", "输入待分类短信后，点击“预测分类”按钮，"))
        self.label_9.setText(_translate("NaiveBayes_Spam_filter", "将使用训练好的模型进行预测，"))
        self.label_10.setText(_translate("NaiveBayes_Spam_filter", "预测结果将在下方文本框中显示。"))
        self.resetpredictButton.setText(_translate("NaiveBayes_Spam_filter", "重置"))
