import sys
import training as train
import test as predict
from PyQt5.QtWidgets import QApplication, QMainWindow
from designer.UI_spam import Ui_NaiveBayes_Spam_filter
from functools import partial

def exit_isclicked():
    """
    退出按钮按下
    """
    sys.exit()

def trainfunc(ui):
    """
    开始训练按钮按下
    """
    ui.infoBrowser.append("正在训练中...")
    train.trainmain(ui)
    ui.infoBrowser.append("训练完成！")

def resettrain(ui):
    """
    训练区域重置按钮按下
    """
    ui.infoBrowser.clear()

def resetpredict(ui):
    """
    预测区域重置按钮按下
    """
    ui.lineEdit.clear()
    ui.resultBrowser.clear()

def predictfunc(ui):
    """
    预测分类按钮按下
    """
    inputstr = ui.lineEdit.text()
    predict.simpleTest(str(inputstr), ui)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_NaiveBayes_Spam_filter()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.exitButton.clicked.connect(exit_isclicked)
    ui.resettrainButton.clicked.connect(partial(resettrain, ui))
    ui.trainButton.clicked.connect(partial(trainfunc, ui))
    ui.resetpredictButton.clicked.connect(partial(resetpredict, ui))
    ui.predictButton.clicked.connect(partial(predictfunc, ui))
    sys.exit(app.exec_())