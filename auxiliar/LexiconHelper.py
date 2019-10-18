"""

Clase para el tratamiento del lexicon de sentimientos

Capacidad para clasificar en positivo y negativo
además de las siguientes emociones:
    ira
    anticipacion
    disgusto
    miedo
    alegria
    tristeza
    sorpresa
    confianza
"""

# Author Adrián Insua Yañez
import csv
import pickle

import re
import pandas as pd
import unicodedata
from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer('spanish')


class LexiconHelper:
    def __init__(self, v=0):
        self.v = v
        self.lexicon = self.__load_lexicon()

    @staticmethod
    def strip_accents(s):
        nkfd_form = unicodedata.normalize('NFKD', s)
        return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')

    def __load_lexicon(self):
        # Carga del lexicon para usar en la clase
        try:
            f1 = open('data/lexicon_parsed.pickle', 'rb')
            lexicon = pickle.load(f1)
            """print(lexicon)
            lexicon = pd.read_csv('./../data/lexicon_parsed.csv', encoding='utf-8')"""
            return lexicon
        except Exception as e:
            lexicon = pd.read_csv('./../data/lexicon.csv', encoding='utf-8')
            lexicon = self.__check_ambiguity(self.__filter_lexicon_binary(lexicon))
            lexicon = self.__stem_lexicon(lexicon)
            self.save_lexicon(lexicon)
            """lexicon = self.__parse_accents(lexicon)
            self.save_lex_df(lexicon)"""
            return lexicon

    def __parse_accents(self, lexicon):
        lex = pd.DataFrame(columns=('word', 'polarity'))
        for l in lexicon.itertuples():
            row_s = self.create_row(l.Index, ['word', 'polarity'],
                                    [self.strip_accents(l.Word), l.Positive+(l.Negative*-1)])
            lex = lex.append(row_s)
        return lex

    @staticmethod
    def __check_ambiguity(lexicon):
        new_lex = {}
        for w in lexicon.itertuples():
            if w.Word in new_lex.keys():
                new_lex.update({w.Word: new_lex[w.Word] + (w.Positive - w.Negative)})
                if new_lex[w.Word] > 1:
                    new_lex.update({w.Word: 1})
                elif new_lex[w.Word] < -1:
                    new_lex.update({w.Word: -1})
            else:
                new_lex.update({w.Word: (w.Positive - w.Negative)})
        return new_lex


    @staticmethod
    def __stem_lexicon(lexicon):
        new_lex = {}
        for w in lexicon:
            stem = stemmer.stem(w)
            if stem in new_lex.keys():
                new_lex.update({stem: new_lex[stem] + lexicon[w]})
            else:
                new_lex.update({stem: lexicon[w]})
        return new_lex

    @staticmethod
    def __filter_lexicon_binary(lexicon):
        lex = lexicon[(lexicon.Positive != 0) | (lexicon.Negative != 0)]
        return lex

    @staticmethod
    def save_lexicon(lexicon):
        f = open('./../data/lexicon_parsed.pickle', 'wb')
        pickle.dump(lexicon, f)
        f.close()

    def save_lex_df(self, lexicon):
        lexicon.to_csv('./../data/lexicon_parsed.csv', index=False, encoding='utf-8')

    @staticmethod
    def create_row(i, titles, data):
        """

        :param i: indice de la fila
        :param titles: Titulos de la fila
        :param data: datos a procesar
        :return: pandas.Series con los datos de la fila

        """
        row = dict(zip(titles, data))
        row_s = pd.Series(row)
        row_s.name = i
        return row_s


if __name__ == "__main__":
    lh = LexiconHelper()

