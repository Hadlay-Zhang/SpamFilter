import AdaboostNavieBayes as boostNaiveBayes
import random
import numpy as np


def trainingAdaboostGetDS(ui,iterateNum=40):
    """
    测试分类的错误率
    :param iterateNum:
    :return:
    """
    filename = './dataset/SMSCollection.txt'
    smsWords, classLables = boostNaiveBayes.loadSMSData(filename)

    # 交叉验证
    testWords = []
    testWordsType = []

    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(smsWords)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(smsWords[randomIndex])
        del (smsWords[randomIndex])
        del (classLables[randomIndex])
    """
    训练阶段，可将选择的vocabularyList也放到整个循环中，以选出
    错误率最低的情况，获取最低错误率的vocabularyList
    """
    vocabularyList = boostNaiveBayes.createVocabularyList(smsWords)
    ui.infoBrowser.append("生成词袋模型完成！")
    print("生成词袋模型完成！")
    trainMarkedWords = boostNaiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    ui.infoBrowser.append("数据标记完成！")
    print("数据标记完成！")
    # 转成array向量
    trainMarkedWords = np.array(trainMarkedWords)
    ui.infoBrowser.append("数据转成矩阵完成！")
    print("数据转成矩阵完成！")
    pWordsSpamicity, pWordsHealthy, pSpam = \
        boostNaiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    DS = np.ones(len(vocabularyList))

    ds_errorRate = {}
    minErrorRate = np.inf
    for i in range(iterateNum):
        errorCount = 0.0
        for j in range(testCount):
            testWordsCount = boostNaiveBayes.setOfWordsToVecTor(vocabularyList, testWords[j])
            ps, ph, smsType = boostNaiveBayes.classify(pWordsSpamicity, pWordsHealthy,
                                                       DS, pSpam, testWordsCount)

            if smsType != testWordsType[j]:
                errorCount += 1
                # alpha = (ph - ps) / ps
                alpha = ps - ph
                # if alpha < 0:  # 原先为spam，预测成ham， ERROR!
                if alpha > 0: # 原先为ham，预测成spam
                    DS[testWordsCount != 0] = np.abs(
                            (DS[testWordsCount != 0] - np.exp(alpha)) / DS[testWordsCount != 0])
                # else:  # 原先为ham，预测成spam，ERROR
                else:  # 原先为spam，预测成ham
                    DS[testWordsCount != 0] = (DS[testWordsCount != 0] + np.exp(alpha)) / DS[testWordsCount != 0]
        DSinfo = "DS: "+str(DS)
        ui.infoBrowser.append(DSinfo)
        print('DS: '+str(DS))
        errorRate = errorCount / testCount
        if errorRate < minErrorRate:
            minErrorRate = errorRate
            ds_errorRate['minErrorRate'] = minErrorRate
            ds_errorRate['DS'] = DS
        errorinfo = "第 " + str(i) + " 轮迭代，错误个数 " + str(errorCount) + " ，错误率 "+ str(errorRate)
        ui.infoBrowser.append(errorinfo)
        print('第 %d 轮迭代，错误个数 %d ，错误率 %f' % (i, errorCount, errorRate))
        if errorRate == 0.0:
            break
    ds_errorRate['vocabularyList'] = vocabularyList
    ds_errorRate['pWordsSpamicity'] = pWordsSpamicity
    ds_errorRate['pWordsHealthy'] = pWordsHealthy
    ds_errorRate['pSpam'] = pSpam
    return ds_errorRate


def trainmain(ui):
    dsErrorRate = trainingAdaboostGetDS(ui)
    # 保存模型训练的信息
    np.savetxt('./train_model/pWordsSpamicity.txt', dsErrorRate['pWordsSpamicity'], delimiter='\t')
    np.savetxt('./train_model/pWordsHealthy.txt', dsErrorRate['pWordsHealthy'], delimiter='\t')
    np.savetxt('./train_model/pSpam.txt', np.array([dsErrorRate['pSpam']]), delimiter='\t')
    np.savetxt('./train_model/trainDS.txt', dsErrorRate['DS'], delimiter='\t')
    np.savetxt('./train_model/trainMinErrorRate.txt', np.array([dsErrorRate['minErrorRate']]), delimiter='\t')
    vocabulary = dsErrorRate['vocabularyList']
    fw = open('./train_model/vocabularyList.txt', 'w')
    for i in range(len(vocabulary)):
        fw.write(vocabulary[i] + '\t')
    fw.flush()
    fw.close()
