import codecs

import stanza

# import stanfordnlp
#               precision    recall  f1-score   support
#
#            B     0.8952    0.8916    0.8934     56882
#            M     0.4835    0.8337    0.6121     11479
#            E     0.8874    0.8839    0.8857     56882
#            S     0.9316    0.7774    0.8476     47490
#
#    micro avg     0.8538    0.8538    0.8538    172733
#    macro avg     0.7994    0.8466    0.8097    172733
# weighted avg     0.8753    0.8538    0.8595    172733

# 百度网盘地址https://pan.baidu.com/s/1Qv3jUDTnRt43VE99qybjyA  提取码wf76  参考放置路径为C:\Users\Administrator\stanza_resources
stanza.download('zh')
nlp = stanza.Pipeline(lang="zh")  # This sets up a default neural pipeline in English

with codecs.open('../plain/pku_test.utf8', 'r', encoding='utf8') as ft:
    with codecs.open('pku_test_stanza.txt', 'w', encoding='utf8') as fj:
        lines = ft.readlines()

        for line in lines:
            line = line.strip()
            # _nlp = nlp(line)
            # print(_nlp.sentences)
            # print(_nlp.entities)
            for s in nlp(line).sentences:
                for w in s.words:
                    fj.write(w.text + "  ")
            fj.write('\n')

MODE = 1

if MODE == 1:
    INPUT = 'pku_test_stanza.txt'
    OUTPUT = 'pku_test_stanza_states.txt'

with codecs.open(OUTPUT, 'w', encoding='utf8') as wf:
    with codecs.open(INPUT, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split(' ')
            state = ""
            for word in words:
                if len(word) == 0 or word == "\r\n":
                    continue
                if len(word) - 2 < 0:
                    state += 'S'
                else:
                    state += "B" + "M" * (len(word) - 2) + "E"
            wf.write(state + "\n")
    print("FIN")
