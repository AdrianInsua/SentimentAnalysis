#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Clase de pre-procesamiento

Abarca stemming, tokenización, filtrado de texto, corrección de negación y eliminación de
nombres propios

El nivel de actuación de la clase se decide según el valor de la variable de preprocesado (pre_level)

"""

# Autor Adrián Insua Yañez
import csv
import re
import pandas as pd
import numpy as np
from string import punctuation

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from auxiliar.EmoticonsHelper import emo_lexicon

# clase para el preprocesamiento del texto
tagger = None
stemmer = SnowballStemmer('spanish')
preposiciones = ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta',
                 'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'via']
articulos = ['el', 'la', 'los', 'las', 'un', 'uno', 'una', 'unos', 'unas', 'lo', 'al', 'del']
pronombres = ['yo', 'tu', 'vos', 'usted', 'el', 'ella', 'ello', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ustedes',
              'ellos', 'ellas', 'mi', 'conmigo', 'ti', 'contigo', 'vos', 'usted', 'sí', 'consigo', 'me', 'te', 'se', 'le',
              'nos', 'os', 'les', 'que']
pronombres_posesivos = ['mio', 'mia', 'mios', 'mias', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas',
                       'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'suyo',
                       'suya', 'suyos', 'suyas']
pronombres_demostrativos = ['este', 'esta', 'esto', 'estos', 'estas', 'ese', 'esa', 'eso', 'esos', 'esas', 'aquel', 'aquella',
                            'aquello', 'aquellos', 'aquellas']
pronombres_indefinidos = ['alguno', 'alguna', 'algunos', 'algunas', 'ninguno', 'ningunas', 'poco', 'poca', 'pocos', 'pocas',
                         'escaso', 'escasa', 'escasos', 'escasas', 'mucho', 'mucha', 'muchos', 'muchas', 'demasiado',
                         'demasiada', 'demasiados', 'demasiadas', 'todo', 'toda', 'todos','todas', 'varios', 'varias',
                         'otro', 'otra', 'otros', 'otras', 'mismo', 'misma', 'mismos', 'mismas', 'tan', 'tanto', 'tanta',
                         'tantos', 'tantas', 'alguien', 'nadie', 'cualquiera', 'cualesquiera', 'quienquiera', 'quienesquiera', 'demas']
verbo_estar = ['estais', 'estan', 'este', 'estes', 'estemos', 'esteis', 'esten', 'estare', 'estaras', 'estara', 'estaremos',
               'estareis', 'estaran', 'estaria', 'estarias', 'estariamos', 'estariais', 'estarian', 'estaba', 'estabas',
               'estabamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron',
               'estuviera', 'estuvieras', 'estuvieramos', 'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviesemos',
               'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad']
verbo_haber = ['he', 'has', 'ha', 'hemos', 'habeis', 'han', 'haya', 'hayas', 'hayamos', 'hayais', 'hayan', 'habre', 'habras',
               'habra', 'habremos', 'habreis', 'habran', 'habria', 'habrias', 'habriamos', 'habriais', 'habrian', 'habia',
               'habias', 'habiamos', 'habiais', 'habian', 'hube', 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron',
               'hubiera', 'hubieras', 'hubieramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiesemos', 'hubieseis',
               'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas']
verbo_ser = ['soy', 'eres', 'es', 'somos', 'sois', 'son', 'sea', 'seas', 'seamos', 'seais', 'sean', 'sere', 'seras', 'sera',
             'seremos', 'sereis', 'seran', 'seria', 'serias', 'seriamos', 'seriais', 'serian', 'era', 'eras', 'eramos', 'erais',
             'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron', 'fuera', 'fueras', 'fueramos', 'fuerais', 'fueran',
             'fuese', 'fueses', 'fuesemos', 'fueseis', 'fuesen']
verbo_sentir = ['sintiendo', 'sentido', 'sentida', 'sentidos', 'sentidas', 'siente', 'sentid']
verbo_tener = ['tengo', 'tienes', 'tiene', 'tenemos', 'teneis', 'tienen', 'tenga', 'tengas', 'tengamos', 'tengais', 'tengan',
               'tendre', 'tendras', 'tendra', 'tendremos', 'tendreis', 'tendran', 'tendria', 'tendrias', 'tendríamos',
               'tendriais', 'tendrian', 'tenia', 'tenias', 'teniamos', 'teniais', 'tenian', 'tuve', 'tuviste', 'tuvo',
               'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuvieramos', 'tuvierais', 'tuvieran', 'tuviese',
               'tuvieses', 'tuviesemos', 'tuvieseis', 'tuviesen', 'teniendo', 'tenido', 'tenida', 'tenidos', 'tenidas',
               'tened']
otros = ['y', 'o', 'pero', 'mas', 'porque', 'como']

spanish_stopwords = []
spanish_stopwords.extend(preposiciones)
spanish_stopwords.extend(articulos)
spanish_stopwords.extend(pronombres)
spanish_stopwords.extend(pronombres_posesivos)
spanish_stopwords.extend(pronombres_demostrativos)
spanish_stopwords.extend(pronombres_indefinidos)
spanish_stopwords.extend(verbo_estar)
spanish_stopwords.extend(verbo_haber)
spanish_stopwords.extend(verbo_ser)
spanish_stopwords.extend(verbo_sentir)
spanish_stopwords.extend(verbo_tener)
spanish_stopwords.extend(otros)
punct = '!"#$%&\'()*+,-./:;<=>?[\\]^_`’‘{|}~«»“”¿¡…'

non_words = list(punct)
non_words.extend(['mencion', 'url', 'hashtag'])
emoticons = emo_lexicon()

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, process=True, stop_words=True, negation=0, repeated_letters=True, v=0):
        self.process = process
        self.stop_words = stop_words
        self.negation = negation
        self.repeated_letters = repeated_letters
        self.v = v

    def set_params(self, process=True, stop_words=True, negation=0, repeated_letters=True, v=0, **args):
        self.process = process
        self.stop_words = stop_words
        self.negation = negation
        self.repeated_letters = repeated_letters
        self.v = v
        self.text = None

    def fit(self, raw_documents, y=None):
        return self

    def transform(self, raw_documents):
        v = self.v
        documents = []
        for text in raw_documents:
            text = text.lower()
            # text = ''.join([negation_parse(c) for c in sent_tokenize(text)])
            if self.process:
                print("Realizando preprocesado nivel 1...") if v >= 1 else None
                text = self.__process_tweet(text)
            text = self.__add_spaces(text)
            if self.stop_words:
                text = ''.join([c + ' ' for c in text.split() if c not in spanish_stopwords])
                text = ''.join([c + ' ' for c in text.split() if c not in non_words])
            if self.negation:
                print("Tratando la negación...") if v >= 1 else None
                text = self.__negacion(text.split(), self.negation, v)
            if self.repeated_letters:
                print("Eliminando letras repetidas....") if v >= 1 else None
                text = self.__replace_two_or_more(text)
            documents.append(text)
        return np.array(documents)

    def fit_transform(self, raw_documents, y=None):
        return self.transform(raw_documents)

    def __filtrar_text(self, text):
        new_text = ''
        for c in text.split():
            non_w_check = [c.__contains__(x) for x in non_words]
            if any(non_w_check):
                for i in range(len(non_w_check)):
                    if non_w_check[i]:
                        c = c.replace(non_words[i], '')
            if len(c) > 0:
                new_text += c + ' '
        return new_text


    # start process_tweet
    def __process_tweet(self, tweet):
        # process the tweets
        # Convert www.* or https?://* to URL
        tweet = re.sub('([\W]xq[\W])|^xq[\W]', ' porque ', tweet)
        tweet = re.sub('([\W]x[\W])|^x[\W]', ' por ', tweet)
        tweet = re.sub('([\W]xara[\W])|^xara[\W]', ' para ', tweet)
        tweet = re.sub('([\W]xa[\W])|^xa[\W]', ' para ', tweet)
        tweet = re.sub('([\W](tk|tq)[\W])|^(tk|tq)[\W]', ' te quiero ', tweet)
        tweet = re.sub('([\W]lol[\W])|^lol[\W]', ' carcajada ', tweet)
        tweet = re.sub('([\W]lmao[\W])|^lmao[\W]', ' risa ', tweet)
        tweet = re.sub('([\W](k|q)[\W])|^(k|q)[\W]', ' que ', tweet)
        tweet = re.sub('([\W]d[\W])|^d[\W]', ' de ', tweet)
        tweet = re.sub('([\W]n[\W])|^n[\W]', ' en ', tweet)
        tweet = re.sub('([\W]dl[\W])|^dl[\W]', ' del ', tweet)
        tweet = re.sub('([\W]bss[\W])|^bss[\W]', ' besos ', tweet)
        tweet = re.sub('([\W]mx[\W])|^mx[\W]', ' mucho ', tweet)
        tweet = re.sub('([\W](tkm|tqm)[\W])|^(tkm|tqm)[\W]', ' te quiero mucho ', tweet)
        tweet = re.sub('([\W](xd|XD|xD)[\W])|^(xd|XD|xD)[\W]', ' risa ', tweet)
        tweet = re.sub('([\W]pls[\W])|^pls[\W]', ' por favor ', tweet)
        tweet = re.sub('([\W]thx[\W])|^thx[\W]', ' gracias ', tweet)
        tweet = re.sub('([\W](bdías|bdias)[\W])|^(bdías|bdias)[\W]', ' buenos días ', tweet)
        tweet = re.sub('([0-9]+)', '', tweet)
        tweet = re.sub('(\xc2\xa0*)', '', tweet)
        tweet = tweet.replace(u'\xa0', u' ')
        tweet = re.sub('(\\\)', '', tweet)

        tweet = re.sub('((www\.[^\s]+)|(https?:[^\s]+))', 'url', tweet)
        # control emoticonos
        text = ''.join(
            [emoticons[c] if c in emoticons else c + ' ' for c in tweet.split()]
        )
        tweet = text
        # Convert @username to AT_USER
        tweet = re.sub('@[^\s]+', ' mencion ', tweet)
        # Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
        # Replace #word with word
        tweet = re.sub('#([^\s]+)', ' hashtag ', tweet)
        # trim
        tweet = tweet.strip('\'"')
        return tweet
        # end


    # start replaceTwoOrMore
    def __replace_two_or_more(self, s):
        # look for 2 or more repetitions of character and replace with the character itself
        pattern = re.compile(r"(.)\1+", re.DOTALL)
        s = pattern.sub(r"\1\1", s)
        return pattern.sub(r"\1\1", s)
        # end


    #start add space inpunctuation
    def __add_spaces(self, s):
        pattern = re.compile(r"([" + punct + "])", re.DOTALL)
        return pattern.sub(r" \1 ", s)

    def __negacion(self, text, th, v):
        new_text = ""
        old_th = th
        for i in range(len(text)):
            if th != old_th:
                th -= 1
            if th == -1:
                new_text += "negado" + text[i] + " "
                th = old_th
            elif text[i] == 'no':
                th -= 1  # Se empieza a descontar hasta llegar al valor 0 de treshold
                new_text += text[i] + " "
            else:
                new_text += text[i] + " "
        print("nuevo texto ", new_text) if v >= 3 else None
        return new_text


    def __process_tags(self, text, v):
        new_text = ""
        for i in range(len(text)):
            if isinstance(text[i], tw.NotTag) is False:
                if text[i][1] != 'NP':
                    # Eliminamos los nombres propios
                    new_text += text[i][0] + " "
        return new_text

def distance_levenshtein(self, str1, str2):
    d=dict()
    for i in range(len(str1)+1):
        d[i]=dict()
        d[i][0]=i
    for i in range(len(str2)+1):
        d[0][i] = i
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            d[i][j] = min(d[i][j-1]+1, d[i-1][j]+1, d[i-1][j-1]+(not str1[i-1] == str2[j-1]))
    return d[len(str1)][len(str2)]


def stemize(text):
    """

    :param text: texto a tokenizar
    :param est: Flag para usar el stemming
    :return:

    Además de la tokenización se centraliza en esta función todo el pre-procesado para que
    devuelva el resultado final al metodo de sklearn utilizado

    """
    tokens = word_tokenize(text)
    # stem
    try:
        stems = __stem_tokens(tokens)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

def tokenize(text):
    """

    :param text: texto a tokenizar
    :param est: Flag para usar el stemming
    :return:

    Además de la tokenización se centraliza en esta función todo el pre-procesado para que
    devuelva el resultado final al metodo de sklearn utilizado

    """
    tokens = word_tokenize(text)
    # stem
    return tokens


def __stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


preprocessor = Preprocessor(v=0)

vectorizerIdf = TfidfVectorizer(
    analyzer='word', lowercase=False,
    stop_words=spanish_stopwords
)

vectorizer = CountVectorizer(
    analyzer='word', lowercase=False,
    stop_words=spanish_stopwords
)


