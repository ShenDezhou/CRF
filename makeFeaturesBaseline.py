import codecs
import pickle
import re
import string

from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

chars = []

with codecs.open('pku_dic/pku_dict.utf8', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        for w in line:
            if w == '\n':
                continue
            else:
                chars.append(w)
print(len(chars))

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
        lines = fa.readlines()
        lines.extend(fb.readlines())
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


MODE = 3

if MODE == 1:
    with codecs.open('plain/pku_training.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('plain/pku_train_states.txt', 'r', encoding='utf8') as fs:
            with codecs.open('model/pku_train_crffeatures.pkl', 'wb') as fx:
                with codecs.open('model/pku_train_crfstates.pkl', 'wb') as fy:
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
    with codecs.open('model/pku_train_crffeatures.pkl', 'rb') as fx:
        with codecs.open('model/pku_train_crfstates.pkl', 'rb') as fy:
            with codecs.open('model/pku_train_crfmodel.pkl', 'wb') as fm:
                bx = fx.read()
                by = fy.read()
                X = pickle.loads(bx)
                y = pickle.loads(by)
                print(X[-1])
                print(y[-1])
                for i in range(len(X)):
                    assert len(X[i]) == len(y[i])
                print('training')
                crf.fit(X, y)
                print('trained')
                sm = pickle.dumps(crf)
                fm.write(sm)
                yp = crf.predict(X)
                print(yp)
                m = metrics.flat_classification_report(
                    y, yp, labels=list("BMES"), digits=4
                )
                print(m)
            # print(pos_tag(sentence))

if MODE == 3:

    with codecs.open('plain/pku_test.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('plain/pku_test_states.txt', 'r', encoding='utf8') as fs:
            with codecs.open('model/pku_train_crfmodel.pkl', 'rb') as fm:
                with codecs.open('baseline/pku_test_crf_states.txt', 'w') as fp:
                    sm = fm.read()
                    crf = pickle.loads(sm)
                    lines = ft.readlines()
                    states = fs.readlines()
                    X = []
                    counter = 0
                    for line in lines:
                        line = line.strip()
                        X.append([getFeaturesDict(line, i) for i in range(len(line))])
                        counter += 1
                        if counter % 100 == 0 and counter != 0:
                            print('.')
                    y = []
                    for state in states:
                        state = state.strip()
                        y.append(list(state))
                    for i in range(len(X)):
                        assert len(X[i]) == len(y[i])
                    yp = crf.predict(X)
                    for sl in yp:
                        for s in sl:
                            fp.write(s)
                        fp.write('\n')
                    m = metrics.flat_classification_report(
                        y, yp, labels=list("BMES"), digits=4
                    )
                    print(m)
