import pandas as pd
from auxiliar.PrintProgress import print_progress

class negation_preprocess:
    def __init__(self):
        self.data_corpus = None
        try:
            self.data_corpus = self.load_corpus('data/neg_sentences.csv')
        except Exception as e:
            print(e)

    def isNaN(self, number):
        return number != number

    def convert_data(self):
        try:
            neg_corpus = self.load_corpus('data/neg_preprocessed.csv')
        except Exception as e:
            neg_corpus = pd.DataFrame(columns=('content', 'polarity'))
            for i, d in self.data_corpus.iterrows():
                text = d.sentence
                if not self.isNaN(text):
                    polarity = 0
                    if d.change == 'yes':
                        polarity = 1
                    elif d.polarity_modifier == 'increment':
                        polarity = 2
                    elif d.polarity_modifier == 'reduction':
                        polarity = 3
                    row_s = self.create_row(i, ['content', 'polarity'],
                                            [text, polarity])
                    neg_corpus = neg_corpus.append(row_s)
            print_progress(i, len(self.data_corpus), prefix='Progreso:', suffix='Completado', barLength=50)
            self.save_corpus(neg_corpus)
        return neg_corpus

    @staticmethod
    def save_corpus(corpus):
        corpus.to_csv('data/neg_preprocessed.csv', encoding='utf-8')

    @staticmethod
    def load_corpus(direction):
        return pd.read_csv(direction, encoding='utf-8')

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
