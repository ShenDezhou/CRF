import codecs
import pickle
import re
import string

from sklearn.model_selection import GroupKFold
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

dicts = []
unidicts = []
predicts = []
sufdicts = []
longdicts = []
puncdicts = []
digitsdicts = []
chidigitsdicts = []
letterdicts = []
otherdicts = []

Thresholds = 0.95


def getTopN(dictlist):
    adict = {}
    for w in dictlist:
        adict[w] = adict.get(w, 0) + 1
    topN = max(adict.values())
    alist = [k for k, v in adict.items() if v >= topN * Thresholds]
    return alist


with codecs.open('pku_dic/pku_training_words.utf8', 'r', encoding='utf8') as fa:
    with codecs.open('pku_dic/pku_test_words.utf8', 'r', encoding='utf8') as fb:
        with codecs.open('pku_dic/contract_words.utf8', 'r', encoding='utf8') as fc:
            lines = fa.readlines()
            lines.extend(fb.readlines())
            lines.extend(fc.readlines())
            lines = [line.strip() for line in lines]
            dicts.extend(lines)
            # uni, pre, suf, long 这四个判断应该依赖外部词典，置信区间为95%，目前没有外部词典，就先用训练集词典来替代
            unidicts.extend([line for line in lines if len(line) == 1 and re.search(u'[\u4e00-\u9fff]', line)])
            predicts.extend([line[0] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
            predicts = getTopN(predicts)
            sufdicts.extend([line[-1] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
            sufdicts = getTopN(sufdicts)
            longdicts.extend([line for line in lines if len(line) > 3 and re.search(u'[\u4e00-\u9fff]', line)])
            puncdicts.extend(string.punctuation)
            puncdicts.extend(list("！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰–‘’‛“”„‟…‧﹏"))
            digitsdicts.extend(string.digits)
            chidigitsdicts.extend(list("零一二三四五六七八九十百千万亿兆〇零壹贰叁肆伍陆柒捌玖拾佰仟萬億兆"))
            letterdicts.extend(string.ascii_letters)

            somedicts = []
            somedicts.extend(unidicts)
            somedicts.extend(predicts)
            somedicts.extend(sufdicts)
            somedicts.extend(longdicts)
            somedicts.extend(puncdicts)
            somedicts.extend(digitsdicts)
            somedicts.extend(chidigitsdicts)
            somedicts.extend(letterdicts)
            otherdicts.extend(set(dicts) - set(somedicts))


def getCharType(ch):
    types = []

    dictofdicts = [puncdicts, digitsdicts, chidigitsdicts, letterdicts, unidicts, predicts, sufdicts]
    for i in range(len(dictofdicts)):
        if ch in dictofdicts[i]:
            types.append(i)
            break

    extradicts = [longdicts, otherdicts]
    for i in range(len(extradicts)):
        for word in extradicts[i]:
            if ch in word:
                types.append(i + len(dictofdicts))
                break
        if len(types) > 0:
            break

    # if TRANSFER_LEARNING and len(types) ==0:
    #     return str(len(dictofdicts)+len(extradicts)-1)

    assert len(types) == 1 or len(types) == 2, "{} {} {}".format(ch, len(types), types)
    # onehot = [0] * (len(dictofdicts) + len(extradicts))
    # for i in types:
    #     onehot[i] = 1

    return str(types[0])


def safea(sentence, i):
    if i < 0:
        return ''
    if i >= len(sentence):
        return ''
    return sentence[i]


def getNgram(sentence, i):
    ngrams = []
    for offset in [-2, -1, 0, 1, 2]:
        ngrams.append(safea(sentence, i + offset))

    for offset in [-2, -1, 0, 1]:
        ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1))

    for offset in [-1, 0]:
        ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1) + safea(sentence, i + offset + 2))
    return ngrams


def getReduplication(sentence, i):
    reduplication = []
    for offset in [-2, -1]:
        if safea(sentence, i) == safea(sentence, i + offset):
            reduplication.append('1')
        else:
            reduplication.append('0')
    return reduplication


def getType(sentence, i):
    types = []
    for offset in [-1, 0, 1]:
        types.append(getCharType(safea(sentence, i + offset)))
    types.append(getCharType(safea(sentence, i + offset - 1)) + getCharType(safea(sentence, i + offset)) + getCharType(
        safea(sentence, i + offset + 1)))
    return types


def getFeatures(sentence, i):
    features = []
    features.extend(getNgram(sentence, i))
    features.extend(getReduplication(sentence, i))
    features.extend(getType(sentence, i))
    return features


def getFeaturesDict(sentence, i):
    features = []
    features.extend(getNgram(sentence, i))
    features.extend(getReduplication(sentence, i))
    features.extend(getType(sentence, i))
    assert len(features) == 17
    featuresdic = dict([(str(j), features[j]) for j in range(len(features))])
    return featuresdic


MODE = 2

if MODE == 1:
    with codecs.open('plain/contract_train.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('plain/contract_train_states.txt', 'r', encoding='utf8') as fs:
            with codecs.open('model/contract_train_crffeatures.pkl', 'wb') as fx:
                with codecs.open('model/contract_train_crfstates.pkl', 'wb') as fy:
                    xlines = ft.readlines()
                    ylines = fs.readlines()
                    X = []
                    y = []

                    print('process X list.')
                    counter = 0
                    for line in xlines:
                        line = line.replace(" ", "").strip()
                        X.append([getFeaturesDict(line, i) for i in range(len(line))])
                        counter += 1
                        if counter % 100 == 0 and counter != 0:
                            print('.')
                    print(len(X))

                    print('process y list.')
                    for line in ylines:
                        line = line.strip()
                        y.append(list(line))
                    print(len(y))

                    print('validate size.')
                    for i in range(len(X)):
                        assert len(X[i]) == len(y[i])

                    print('output to file.')
                    sX = pickle.dumps(X)
                    fx.write(sX)
                    sy = pickle.dumps(y)
                    fy.write(sy)

if MODE == 2:
    crf = CRF()
    with codecs.open('model/contract_train_crffeatures.pkl', 'rb') as fx:
        with codecs.open('model/contract_train_crfstates.pkl', 'rb') as fy:
            with codecs.open('model/contract_train_crfmodel.pkl', 'wb') as fm:
                with codecs.open('plain/contract_train_group.utf8', 'r') as fg:
                    with codecs.open('plain/contract_train_group_log.utf8', 'w') as fl:
                        groups = fg.readlines()
                        groupKfold = GroupKFold(n_splits=10)
                        bx = fx.read()
                        by = fy.read()
                        X = pickle.loads(bx)
                        y = pickle.loads(by)
                        for i in range(len(X)):
                            assert len(X[i]) == len(y[i])
                        index = 0
                        for train, test in groupKfold.split(X, y, groups=groups):
                            print(index)
                            index += 1
                            gX = [X[i] for i in train]
                            gy = [y[i] for i in train]
                            tX = [X[i] for i in test]
                            ty = [y[i] for i in test]
                            print(gX[-1])
                            print(gy[-1])
                            print('training')
                            crf.fit(gX, gy)
                            print('trained')
                            sm = pickle.dumps(crf)
                            fm.write(sm)

                            yp = crf.predict(tX)
                            print(yp)
                            m = metrics.flat_classification_report(
                                ty, yp, labels=list("BMES"), digits=4
                            )
                            print(m)
                            fl.write("\n\n" + str(index) + "\n")
                            fl.write(str(train))
                            fl.write("\n")
                            fl.write(str(test))
                            fl.write("\n")
                            fl.write(m)
