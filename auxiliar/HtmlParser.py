import pickle
import requests
from bs4 import BeautifulSoup
import pandas as pd
import math
from auxiliar.PrintProgress import print_progress

PARSER = "lxml"
MAX = 10000


class HtmlParser():
    def __init__(self, paginas, link, v):
        self.v = v
        self.paginas = paginas
        self.link = link
        self.p = 0
        self.ne = 0
        self.n = 0
        self.corpus = pd.DataFrame(columns=('content', 'polarity'))
        self.rename_dict = {
            '1': 'N+',
            '2': 'N',
            '3': 'NEU',
            '4': 'P',
            '5': 'P+'
        }

    def get_corpus(self):
        try:
            self.corpus = self.load_data()
        except Exception as e:
            print(e)
            link = self.link
            j = 0
            i = 1
            while True:
                if i == self.paginas:
                    self.save_data(self.corpus)
                    break
                i += 1
                data = self.url_opener(link, PARSER)
                if data is not None:
                    res = self.page_parser(data)
                    for d in res:
                        if len(d['texto']) > 0:
                            row_s = self.create_row(j, ['content', 'polarity'],
                                                    [d['texto'], d['nota']])
                            self.corpus = self.corpus.append(row_s)
                            j += 1

                    print("n: %i  neu: %i  p: %i" % (self.n, self.ne, self.p))
                    if MAX > 0:
                        if self.n >= MAX and self.ne >= MAX and self.p >= MAX:
                            break
                    # Buscamos la siguiente pagina
                    spans = data.find_all('span', {'class': 'peli'})
                    aux = spans[len(spans)-1].find('a')  # El link de siguiente página está siempre al final
                    if aux is None:
                        break
                    link = self.link+aux['href']
            self.save_data(self.corpus)
        return self.corpus

    @staticmethod
    def url_opener(url, parser):
        try:
            r = requests.get(url)
            encoding = r.encoding if 'charset' in r.headers.get('content-typ', '').lower() else None
            soup = BeautifulSoup(r.content, parser, from_encoding=encoding)
            return soup
        except Exception as e:
            print(e)
            return None

    def page_parser(self, b_soup):
        data = []
        lista = b_soup.find('div', {'id': 'g2'})
        elementos = lista.find_all('li')
        long = len(elementos)
        print("Número de elementos: %i" % long) if self.v >= 2 else None
        for i, el in enumerate(elementos):
            print_progress(i+1, long, prefix='Progreso:', suffix='Completado', barLength=50) if self.v >= 1 \
                else None
            link = el.find('a', {'class': 'mediana2'})
            val = el.find('a', {'class': None})
            nota = val['href'][val['href'].index('=') + 1:]
            try:
                if MAX > 0:
                    if int(nota) > 3:
                        if self.p > MAX:
                            continue
                    elif int(nota) < 3:
                        if self.n > MAX:
                            continue
                    else:
                        if self.ne > MAX:
                            continue

                # Accedemos a la critica para guardar el texto
                text_soup = self.url_opener(link['href'], PARSER)
                texto = text_soup.find('span', {'class': 'mediana'}).text.replace('\\n', '')
                data.append({'texto': texto, 'nota': nota})
                if int(nota) > 3:
                    self.p += 1
                elif int(nota) < 3:
                    self.n += 1
                else:
                    self.ne += 1
            except Exception as e:
                print(e)
        return data

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

    @staticmethod
    def save_data(corpus):
        corpus.to_csv('data/pelis.csv', index=False, encoding='utf-8')

    @staticmethod
    def load_data():
        return pd.read_csv('data/pelis.csv', encoding='utf-8')

