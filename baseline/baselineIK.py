import codecs

import elasticsearch

# es = elasticsearch.Elasticsearch([{"host": "localhost", "port": 9200}])
# body = {'text': "主体身份位高权重。犯罪嫌疑人中担任董事长、厂长、经理、党委书记职务的企业高层领导约占67％，负责销售、掌管财务的业务员、部门经理约占33％。", 'analyzer': "ik_smart"}
# res = es.indices.analyze(index="cbooo_movie_v3", body=body)
# print(res)
# words = [w['token'] for w in res['tokens']]
# print(words)
# exit(1)

es = elasticsearch.Elasticsearch([{"host": "localhost", "port": 9200}])


def getIKStatesAndWords(sentence):
    body = {'text': "", 'analyzer': "ik_smart"}
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
    with codecs.open('pku_test_ik.txt', 'w', encoding='utf8') as fj:
        with codecs.open('pku_test_ik_states.txt', 'w', encoding='utf8') as fs:
            lines = ft.readlines()
            for line in lines:
                line = line.strip()
                states, words = getIKStatesAndWords(line)
                fj.write(words)
                fj.write('\n')
                fs.write(states)
                fs.write('\n')
                print('.')
