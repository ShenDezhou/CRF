import codecs
import os

MODE = 1
if MODE == 1:
    files = [os.path.join('contract', f) for f in os.listdir('contract') if os.path.isfile(os.path.join('contract', f))]
    with codecs.open('contract_train.utf8', 'w', encoding='utf8') as fw:
        with codecs.open('plain/contract_train_group.utf8', 'w', encoding='utf8') as fg:
            counter = 0
            for f in files:
                with codecs.open(f, 'r', encoding='utf8') as fr:
                    lines = fr.readlines()
                    fw.writelines(lines)
                    for _ in range(len(lines)):
                        fg.write(str(counter))
                        fg.write('\n')
                    counter += 1
            print('FIN')

if MODE == 2:
    INPUT = 'plain/contract_train.utf8'
    OUTPUT = 'plain/contract_train_states.txt'

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

if MODE == 3:
    CORPUS = 'plain/contract_train.utf8'
    DICOUTPUT = 'pku_dic/contract_words.utf8'
    CHAROUTPUT = 'pku_dic/contract_dic.utf8'
    with codecs.open(CHAROUTPUT, 'w', encoding='utf8') as cf:
        with codecs.open(DICOUTPUT, 'w', encoding='utf8') as wf:
            with codecs.open(CORPUS, 'r', encoding='utf8') as rf:
                words = []
                lines = rf.readlines()
                for line in lines:
                    line = line.strip()
                    ws = line.split(' ')
                    ws = [w for w in ws if len(w) != 0]
                    words.extend(ws)
                words = list(set(words))
                words.sort()
                for word in words:
                    wf.write(word)
                    wf.write('\n')

                chars = []
                for w in words:
                    w = w.strip()
                    chars.extend(list(w))
                chars = list(set(chars))
                chars.sort()
                for c in chars:
                    cf.write(c)
