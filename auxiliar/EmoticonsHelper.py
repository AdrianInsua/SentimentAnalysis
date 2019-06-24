from mako.filters import trim
import os

sent = {
    '1': 'bueno',
    '0': 'neutro',
    '-1': 'malo'
}


def emo_lexicon():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/EmoticonSentimentLexicon.txt')
    f = open(filename, 'r', encoding='utf-8')

    output_dict = {}
    for line in f:
        emoti = trim(line[:(len(line) - 3)])
        pol = trim(line[len(line)-3:])
        output_dict.update({emoti: sent[pol]})
    return output_dict
