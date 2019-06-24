"""
Clase auxiliar para el parseo del xml del corpus de tweets del TASS

"""

# Autor Adrián Insua Yañez

import pandas as pd


class DataFrameHelper:
    def __init__(self, xml=None, out='', v=1):
        """

        :param xml: children de lxml root
        :param out: Dirección en la que se guardará el csv. String
        :param v: Nivel de verbose

        """
        self.xml = xml
        self.v = v
        self.out = out

    def convert(self):
        """

        :return: Persistencia del csv en disco

        Itera la colección de nodos del xml y guarda el resultado en disco

        """
        tweet_corpus = self.__iter_docs()
        return tweet_corpus

    def save_csv(self, data, direccion):
        print("Grabando csv del corpus...") if self.v >= 1 else None
        data.to_csv(direccion, index=False, encoding='utf-8')

    def __iter_docs(self):
        """

        :return: Corpus compuesto en una instancia pandas.Dataframe

         Itera la colección agregando solo los valores no nulos o neutros
         y de los cuales hay concordancia o aceptación en la valoración

        """
        tweet_corpus = pd.DataFrame(columns=('content', 'polarity'))
        long = len(self.xml)
        for i in range(0, long):
            tweet = self.xml[i]
            if tweet.content.text is not None \
                    and tweet.sentiments.polarity.value.text not in ['NONE'] \
                    and tweet.sentiments.polarity.type.text == 'AGREEMENT':

                row_s = self.create_row(i, ['content', 'polarity'],
                                        [tweet.content.text, tweet.sentiments.polarity.value.text])
                tweet_corpus = tweet_corpus.append(row_s)
        return tweet_corpus

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
