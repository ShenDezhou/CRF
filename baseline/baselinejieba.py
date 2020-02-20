import codecs

import jieba

with codecs.open('../plain/pku_test.utf8', 'r', encoding='utf8') as ft:
    with codecs.open('pku_test_jieba.txt', 'w', encoding='utf8') as fj:
        lines = ft.readlines()
        for line in lines:
            line = line.strip()
            words = jieba.lcut(line)
            for w in words:
                fj.write(w + "  ")
            fj.write('\n')

MODE = 1

if MODE == 1:
    INPUT = 'pku_test_jieba.txt'
    OUTPUT = 'pku_test_jieba_states.txt'

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
