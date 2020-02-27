# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
import codecs
import re
import string

import numpy
from keras import regularizers
from keras.layers import Dense, Embedding, LSTM, Dropout, Input, Bidirectional
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

#               precision    recall  f1-score   support
#
#            B     0.3381    0.3528    0.3453     56882
#            M     0.0000    0.0000    0.0000     11479
#            E     0.3354    0.4322    0.3777     56882
#            S     0.3145    0.2655    0.2880     47490
#
#     accuracy                         0.3315    172733
#    macro avg     0.2470    0.2626    0.2527    172733
# weighted avg     0.3083    0.3315    0.3173    172733
# mean_squared_error 0.2586491465518497
# mean_absolute_error 0.27396197698378544
# mean_absolute_percentage_error 0.3323864857505891
# mean_squared_logarithmic_error 0.2666326968685906
# squared_hinge 0.2827528866772688
# hinge 0.27436352076398335
# categorical_crossentropy 0.3050300775957548
# binary_crossentropy 0.7499999871882543
# kullback_leibler_divergence 0.30747676168440974
# poisson 0.2897763648871911
# cosine_proximity 0.3213321868358391
#
#
# sgd 0.27380688950156684
# rmsprop 0.4363407859974404
# adagrad 0.5028908227192664
# adadelta 0.3134481079882679
# adam 0.342444794579377
# adamax 0.36860069757644914
# nadam 0.39635284171196516
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

chars = []

with codecs.open('pku_dic/pku_dict.utf8', 'r', encoding='utf8') as f:
    # with codecs.open('pku_diccontract_dict.utf8', 'r', encoding='utf8') as fc:
    lines = f.readlines()
    # lines.extend(fc.readlines())
    for line in lines:
        for w in line:
            if w == '\n':
                continue
            else:
                chars.append(w)
print(len(chars))

rxdict = dict(zip(chars, range(1, 1 + len(chars))))
rxdict['\n'] = 0

rydict = dict(zip(list("BMES"), range(len("BMES"))))


def getNgram(sentence, i):
    ngrams = []
    ch = sentence[i]
    ngrams.append(rxdict[ch])
    return ngrams


def getFeaturesDict(sentence, i):
    features = []
    features.extend(getNgram(sentence, i))
    assert len(features) == 1
    # featuresdic = dict([(str(j), features[j]) for j in range(len(features))])
    # return featuresdic
    return features


batch_size = 1024
maxlen = 1019
nFeatures = 1
word_size = 100
Hidden = 50
Regularization = 1e-4
Dropoutrate = 0.2
learningrate = 0.2
Marginlossdiscount = 0.2
nState = 4
EPOCHS = 1

modeldic = {}
scoredic = {}

counter = 0
for loss in ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
             "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_crossentropy",
             "sparse_categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence", "poisson",
             "cosine_proximity"]:
    sequence = Input(shape=(maxlen,))
    dropout = Dropout(rate=Dropoutrate)(sequence)
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(dropout)
    blstm = Bidirectional(LSTM(Hidden, return_sequences=True), merge_mode='sum')(embedded)
    # lstm = LSTM(Hidden, return_sequences=True)(embedded)
    dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(blstm)
    model = Model(input=sequence, output=dense)
    # model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=["accuracy"])
    # adagrad = Adagrad(lr=learningrate)
    model.compile(loss=loss, optimizer='sgd', metrics=["accuracy"])
    modeldic[loss] = model
    counter += 1
    model.summary()

for optimizer in ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']:
    sequence = Input(shape=(maxlen,))
    dropout = Dropout(rate=Dropoutrate)(sequence)
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(dropout)
    blstm = Bidirectional(LSTM(Hidden, return_sequences=True), merge_mode='sum')(embedded)
    # lstm = LSTM(Hidden, return_sequences=True)(embedded)
    dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(blstm)
    model = Model(input=sequence, output=dense)
    # model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=["accuracy"])
    # adagrad = Adagrad(lr=learningrate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy"])
    # model.save(r"H:\私人文件\008代码及系统\021字典和Viterbi中文分词\elasticsearch-analysis-hanlp\src\main\resources\lstm%d.h5"%counter)
    modeldic[optimizer] = model
    counter += 1
    model.summary()

MODE = 1

if MODE == 1:
    with codecs.open('plain/pku_training.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('plain/pku_train_states.txt', 'r', encoding='utf8') as fs:
            xlines = ft.readlines()
            ylines = fs.readlines()
            X = []
            y = []

            print('process X list.')
            counter = 0
            for line in xlines:
                line = line.replace(" ", "").strip()
                # X.append([getFeaturesDict(line, i) for i in range(len(line))])
                X.append([rxdict.get(e, 0) for e in list(line)])
                counter += 1
                if counter % 10000 == 0 and counter != 0:
                    print('.')
            print(len(X))
            X = pad_sequences(X, maxlen=maxlen, padding='pre', value=0)
            print(len(X), X.shape)

            print('process y list.')
            for line in ylines:
                line = line.strip()
                line = [rydict[s] for s in line]
                sline = numpy.zeros((len(line), len("BMES")), dtype=int)
                for g in range(len(line)):
                    sline[g, line[g]] = 1
                y.append(sline)
            print(len(y))
            y = pad_sequences(y, maxlen=maxlen, padding='pre', value=3)
            print(len(y), y.shape)
            for key, model in modeldic.items():
                try:
                    history = model.fit(X, y, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)
                    # scores = model.evaluate(X, y, verbose=0)
                    scoredic[key] = history.history['acc'][-1]
                    print(key, scoredic[key])
                    model.save("keras/lstm-%s.h5" % key)
                except:
                    print(key, "has Error")

            print(scoredic)
            print('FIN')

if MODE == 2:
    STATES = list("BMES")
    with codecs.open('plain/pku_test.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('baseline/pku_test_lstm_states.txt', 'w', encoding='utf8') as fl:
            model = load_model("keras/lstm.h5")
            model.summary()

            xlines = ft.readlines()
            X = []
            print('process X list.')
            counter = 0
            for line in xlines:
                line = line.replace(" ", "").strip()
                # X.append([getFeaturesDict(line, i) for i in range(len(line))])
                X.append([rxdict.get(e, 0) for e in list(line)])
                counter += 1
                if counter % 1000 == 0 and counter != 0:
                    print('.')
            print(len(X))
            X = pad_sequences(X, maxlen=maxlen, padding='pre', value=0)
            print(len(X), X.shape)
            yp = model.predict(X)
            print(yp.shape)
            for i in range(yp.shape[0]):
                sl = yp[i]
                lens = len(xlines[i].strip())
                for s in sl[-lens:]:
                    i = numpy.argmax(s)
                    fl.write(STATES[i])
                fl.write('\n')
            print('FIN')
            # for sl in yp:
            #     for s in sl:
            #         i = numpy.argmax(s)
            #         fl.write(STATES[i])
            #     fl.write('\n')
            # print('FIN')
