import codecs

import pkuseg

# baseline
#               precision    recall  f1-score   support
#
#            B     0.9111    0.9573    0.9336     56882
#            M     0.7630    0.8147    0.7880     11479
#            E     0.9200    0.9667    0.9428     56882
#            S     0.9791    0.8440    0.9065     47490
#
#    micro avg     0.9198    0.9198    0.9198    172733
#    macro avg     0.8933    0.8957    0.8927    172733
# weighted avg     0.9229    0.9198    0.9195    172733


pkuseg = pkuseg.pkuseg(model_name='web')

with codecs.open('../plain/pku_test.utf8', 'r', encoding='utf8') as ft:
    with codecs.open('pku_test_pkuseg.txt', 'w', encoding='utf8') as fj:
        lines = ft.readlines()
        for line in lines:
            line = line.strip()
            words = pkuseg.cut(line)
            for w in words:
                fj.write(w + "  ")
            fj.write('\n')

MODE = 1

if MODE == 1:
    INPUT = 'pku_test_pkuseg.txt'
    OUTPUT = 'pku_test_pkuseg_states.txt'

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
