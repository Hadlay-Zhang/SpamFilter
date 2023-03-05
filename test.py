import numpy as np
import AdaboostNavieBayes as boostNaiveBayes

def processinput(input):
    """
    加载输入短信
    :param input:
    :return:
    """
    Words = []
    # 切分文本
    splitwords = boostNaiveBayes.textParser(input)
    Words.append(splitwords)
    return Words

def getTrainAdaboostInfo():
    """
    获取训练算法阶段的DS和minErrorRate信息
    :return:
    """
    trainDS = np.loadtxt('./train_model/trainDS.txt', delimiter='\t')
    trainMinErrorRate = np.loadtxt('./train_model/trainMinErrorRate.txt', delimiter='\t')
    vocabularyList = boostNaiveBayes.getVocabularyList('./train_model/vocabularyList.txt')
    pWordsSpamicity = np.loadtxt('./train_model/pWordsSpamicity.txt', delimiter='\t')
    pWordsHealthy = np.loadtxt('./train_model/pWordsHealthy.txt', delimiter='\t')
    pSpam = np.loadtxt('./train_model/pSpam.txt', delimiter='\t')
    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, trainMinErrorRate, trainDS


def simpleTest(input,ui):
    # 若输入文本框为空，则不进行预测操作，继续等待输入
    if (len(input) == 0):
        return
    # 加载训练好的模型信息
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, trainMinErrorRate, trainDS = \
        getTrainAdaboostInfo()
    smsWords = processinput(input)
    print(smsWords[0])
    testWordsMarkedArray = \
        boostNaiveBayes.setOfWordsToVecTor(vocabularyList, smsWords[0])
    ps, ph, smsType = boostNaiveBayes.classify(
            pWordsSpamicity, pWordsHealthy, trainDS, pSpam, testWordsMarkedArray)
    if (smsType == 1):
        ui.resultBrowser.clear()
        ui.resultBrowser.append('是垃圾短信！')
        #print('是垃圾短信！')
    else:
        ui.resultBrowser.clear()
        ui.resultBrowser.append('是普通短信！')
        #print('是普通短信！')
