import codecs

from sklearn_crfsuite import metrics

MODE = 4

GOLD = '../plain/pku_test_states.txt'

if MODE == 1:
    TEST = 'pku_test_jieba_states.txt'

if MODE == 2:
    TEST = 'pku_test_ik_states.txt'

if MODE == 3:
    TEST = 'pku_test_hmmstate.utf8'

if MODE == 4:
    TEST = 'pku_test_crf_states.txt'

with codecs.open(TEST, 'r', encoding='utf8') as fj:
    with codecs.open(GOLD, 'r', encoding='utf8') as fg:
        jstates = fj.readlines()
        states = fg.readlines()
        y = []
        for state in states:
            state = state.strip()
            y.append(list(state))
        yp = []
        for jstate in jstates:
            jstate = jstate.strip()
            yp.append(list(jstate))
        for i in range(len(y)):
            assert len(yp[i]) == len(y[i])
        m = metrics.flat_classification_report(
            y, yp, labels=list("BMES"), digits=4
        )
        print(m)
