import codecs

import elasticsearch

# 1.smart with no config. comment this config #root=C:/dev/data-for-1.7/
#    macro avg     0.8552    0.8926    0.8674    172733
# weighted avg     0.9047    0.8966    0.8976    172733
# 2.crf
#    macro avg     0.8395    0.9028    0.8539    172733
# weighted avg     0.9216    0.8954    0.9032    172733
# 3.per
#    macro avg     0.8367    0.9004    0.8503    172733
# weighted avg     0.9208    0.8928    0.9012    172733
# 4.nlp
#    macro avg     0.8023    0.8658    0.8096    172733
# weighted avg     0.8908    0.8528    0.8636    172733

es = elasticsearch.Elasticsearch([{"host": "localhost", "port": 9200}])

MODE = 5
if MODE == 1:
    analyzer = "hanlp_smart"

if MODE == 2:
    analyzer = "hanlp_crf"

if MODE == 3:
    analyzer = "hanlp_per"

if MODE == 4:
    analyzer = "hanlp_nlp"

if MODE == 5:
    analyzer = "hanlp_a"


def getHanLPStatesAndWords(sentence):
    body = {'text': "", 'analyzer': analyzer}
    body['text'] = sentence
    res = es.indices.analyze(index="cbooo_movie_v3", body=body)
    tokens = res['tokens']
    index = 0
    sentencep = ""
    res = ""
    for w in tokens:
        while index < w['start_offset']:
            res += 'S' * (w['start_offset'] - index)
            sentencep += "__"
            index = w['start_offset']
        if len(w['token']) == 1:
            res += 'S'
        else:
            res += 'B' + 'M' * (len(w['token']) - 2) + 'E'
        sentencep += w['token'] + "  "
        index += len(w['token'])

    while index < len(sentence):
        res += 'S' * (len(sentence) - index)
        sentencep += "__"
        index = len(sentence)

    assert len(res) == len(sentence), (res, sentence)
    return res, sentencep


with codecs.open('../plain/pku_test.utf8', 'r', encoding='utf8') as ft:
    with codecs.open('pku_test_' + analyzer.replace("_", "") + '.txt', 'w', encoding='utf8') as fj:
        with codecs.open('pku_test_' + analyzer.replace("_", "") + '_states.txt', 'w', encoding='utf8') as fs:
            lines = ft.readlines()
            for line in lines:
                line = line.strip()
                states, words = getHanLPStatesAndWords(line)
                fj.write(words)
                fj.write('\n')
                fs.write(states)
                fs.write('\n')
                print('.')
