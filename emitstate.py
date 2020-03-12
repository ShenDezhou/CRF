import codecs

INPUT = 'plain/pku_test.utf8'
STATE = 'baseline/pku_test_pkuseg_states.txt'
OUTPUT = "plain/pku_test_pkuseg.utf8"

statenames = "BMES"
with codecs.open(OUTPUT, 'w', encoding='utf8') as wf:
    with codecs.open(INPUT, 'r', encoding='utf8') as fr:
        with codecs.open(STATE, 'r', encoding='utf8') as fs:
            statelines = fs.readlines()
            sentences = fr.readlines()
            for linenumber in range(len(statelines)):
                stateline = statelines[linenumber].strip()
                sentence = sentences[linenumber].strip()
                if len(stateline) != len(sentence):
                    print(linenumber)
                for i in range(len(stateline)):
                    wf.write(sentence[i])
                    if stateline[i] == 'E' or stateline[i] == 'S':
                        wf.write("  ")
                wf.write('\n')
    print("FIN")
