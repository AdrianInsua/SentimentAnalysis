import re
import inflect

import unicodedata

from aux.LexiconHelper import LexiconHelper
from nltk.tokenize import sent_tokenize
import treetagger.treetaggerwrapper as tw
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from string import punctuation
from aux.EmoticonsHelper import emo_lexicon

TRESHOLD = 2

stemmer = SnowballStemmer('spanish')
emoticons = emo_lexicon()
tree = tw.TreeTagger(TAGLANG="es")
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
pronom_sufij = ['me', 'te', 'se', 'nos', 'os', 'se']
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
otros = ['y', 'o', 'pero', 'porque', 'como']

abreviaturas = {'xq': 'porque', 'tmb': 'tambien', 'q': 'que', 'd': 'de', 'dl': 'del', 'bdias': 'buenos dias', 'bdías': 'buenos días'}

spanish_stopwords = []
spanish_stopwords.extend(preposiciones)
spanish_stopwords.extend(articulos)
spanish_stopwords.extend(pronombres)
spanish_stopwords.extend(pronombres_posesivos)
spanish_stopwords.extend(pronombres_demostrativos)
spanish_stopwords.extend(verbo_estar)
spanish_stopwords.extend(verbo_haber)
spanish_stopwords.extend(verbo_ser)
spanish_stopwords.extend(verbo_sentir)
spanish_stopwords.extend(verbo_tener)
spanish_stopwords.extend(otros)

non_words = list(punctuation)
non_words.extend(['¿', '¡', '!', '?', '...', ';', ','])
non_words.extend(['MENCION', 'URL', 'NP'])
non_words.extend(map(str, range(10)))


adv_aum = {'muy': 2, 'tod': 0, 'mas': 1.5, 'inclus': 1, 'tambien': 1, 'demasi': 1, 'much': 0.9, 'aumentar': 1.5, 'gran': 2}

adv_dim = {'men': -1.5, 'casi': 0.1, 'nad': -2, 'medi': 0.5, 'bastant': 0.75, 'poc': 0.2, 'algo': 0.4}

adverbios_cant = {'muy': 2, 'casi': 0.1, 'tod': 1, 'medi': 0.5, 'bastant': 0.75,
                  'mas': 1.5, 'men': -1.5, 'ademas': 1, 'inclus': 1, 'tambien': 1}

indef = {'nad': -2, 'poc': 0.25, 'much': 2, 'algo': 0.4, 'demasi': 2, 'tant': 1, 'algun': 1, 'ningún': 1}

adv_modo = {'bien': 1, 'mal': -1}


class PreLexProcess:
    def __init__(self, v=0):
        self.lex = LexiconHelper(v).lexicon
        self.inf = inflect.engine()
        self.v = v
        self.vocab_dict = {}
        self.vocab = []

    def process_data(self, data):
        vocab = {}
        lista = []
        for d in data.itertuples():
            text = self.process_tweet(self.__replace_two_or_more(d.content.lower()))
            lista.append(([w.lemma for w in tw.make_tags(tree.tag_text(text.split()))
                           if not any([w.pos.startswith(v) for v in
                                       ['NEG', 'ACRNM', 'CSUB', 'PREP', 'PE', 'PPO', 'QU']])], d.polarity))
        for l in lista:
            for w in l:
                if w.lemma not in vocab.keys():
                    pols = []
                    l_lista = len(lista)
                    for i in range(0, 1):
                        pols.append(len([c for c in lista if w.lemma in c[0] and c[1] == i])/l_lista)
                    vocab.update({w.lemma: pols})
        self.vocab_dict = vocab
        print(self.vocab_dict)
        self.vocab = vocab.keys()

    def process_features(self, text):
        print(text) if self.v >= 2 else None
        text = self.__filtrar_text(self.__process_tweet(self.__replace_two_or_more(text.lower())))
        text = ''.join([c + ' ' for c in text.split() if c not in spanish_stopwords])
        sents = sent_tokenize(text)
        t_word_pol = [0, 0]
        t_adv = [0, 0, 0, 0]
        for s in sents:
            sub_sent = s.split(',')
            s_word_pol = [0, 0]
            s_adv = [0, 0, 0, 0]
            for ss in sub_sent:
                ts = tw.make_tags(tree.tag_text(ss))
                len_w = len(ts)
                print(ss) if self.v >= 2 else None
                w_cant = self.__process_prob_pos_neg(ts)
                print("w_cant ", w_cant) if self.v >= 2 else None
                print("n. palabras ", sum(w_cant)) if self.v >= 2 else None
                adv_cant = self.__process_prob_aum_dim(ts)
                p_word_pol = [x / sum(w_cant) if sum(w_cant) > 0 else x for x in w_cant]
                print("prob: ", p_word_pol) if self.v >= 2 else None
                w_cant_div = [1 if n == 0 else n for n in w_cant]
                adv_cant = [x / y for n in adv_cant for x, y in zip(n, w_cant_div)]
                s_word_pol = [x + y for x, y in zip(s_word_pol, p_word_pol)]
                s_adv = [x + y for x, y in zip(s_adv, adv_cant)]
            len_ss = len(sub_sent)
            s_word_pol = [x / len_ss for x in s_word_pol]
            s_adv = [x / len_ss for x in s_adv]
            t_word_pol = [x + y for x, y in zip(t_word_pol, s_word_pol)]
            t_adv = [x + y for x, y in zip(t_adv, s_adv)]
        len_sent = len(sents)
        t_word_pol = [x / len_sent for x in t_word_pol]
        t_adv = [x / len_sent for x in t_adv]
        return t_word_pol

    def __process_prob_pos_neg(self, ts):
        len_w = len(ts)
        n_pos, n_neg = 0, 0
        for i in range(len_w):
            if isinstance(ts[i], tw.NotTag) is False:
                if ts[i].pos == 'ADV' and ts[i].word in adv_modo.keys():
                    valor = adv_modo[ts[i].word]
                    if valor > 0:
                        n_pos += 1
                    elif valor < 0:
                        n_neg +=1
                if not any([ts[i].pos.startswith(v) for v in ['NEG', 'ACRNM', 'CSUB', 'PREP', 'PE', 'PPO', 'QU']]):
                    if ts[i].pos.startswith('V'):
                        if ts[i].lemma != ts[i].word:
                            w = ts[i].lemma
                        else:
                            w = re.sub("(se)|(me)|(te)|(le)|(nos)|(os)$", '', ts[i].word)
                    else:
                        w = ts[i].lemma
                    if w in self.lex.keys():
                        print("Palabra relevante %s se busca: %s" % (ts[i], w)) if self.v >= 2 else None
                        valor = self.lex[w]
                        print(valor)
                        if valor > 0:
                            n_pos += 1
                        elif valor < 0:
                            n_neg += 1
        return [n_pos, n_neg]

    def __process_prob_aum_dim(self, ts):
        n_aum_pos, n_dim_pos, n_aum_neg, n_dim_neg = 0, 0, 0, 0
        n_grams = ngrams(ts, 2)
        for n_g in n_grams:
            if isinstance(n_g[0], tw.NotTag) is False:
                if n_g[0].pos == 'ADV' or n_g[0].pos == 'QU':
                    w = [stemmer.stem(n.word) for n in n_g]
                    valor = 0
                    if (not any([n_g[1].pos.startswith(v) for v in ['NEG', 'ACRNM', 'VL', 'CSUBI', 'PREP']]))\
                            and w[1] in self.lex.keys():
                        valor = self.lex[w[1]]
                    elif w[1] in adv_modo:
                        valor = adv_modo[w[1]]
                    print(valor) if self.v >= 2 else None
                    if valor == 1:
                        if w[0] in adv_aum:
                            n_aum_pos += adv_aum[w[0]]
                        elif w[0] in adv_dim:
                            n_dim_pos += adv_dim[w[0]]
                    elif valor == -1:
                        if w[0] in adv_aum:
                            n_aum_neg += adv_aum[w[0]]
                        elif w[0] in adv_dim:
                            n_dim_neg += adv_dim[w[0]]
                    print([n_aum_pos, n_aum_neg], [n_dim_pos, n_dim_neg]) if self.v >= 2 else None
        return [[n_aum_pos, n_aum_neg], [n_dim_pos, n_dim_neg]]

    # start process_tweet
    @staticmethod
    def __process_tweet(tweet):
        # Convert www.* or https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'URL', tweet)
        # control emoticonos
        tweet = re.sub('['+punctuation+']+', ' ', tweet)
        text = ''
        for c in tweet.split():
            if c in abreviaturas.keys():
                text += abreviaturas[c] + " "
            else:
                text += emoticons[c] + " " if c in emoticons else c + " "
        tweet = text
        # Convert @username to AT_USER
        tweet = re.sub('@[^\s]+', ' MENCION ', tweet)
        # Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
        # Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # trim
        tweet = tweet.strip('\'"')
        return tweet
        # end

    # start replaceTwoOrMore
    @staticmethod
    def __replace_two_or_more(s):
        # look for 2 or more repetitions of character and replace with the character itself
        pattern = re.compile(r"([^RrLlcC0-9])\1+", re.DOTALL)
        return pattern.sub(r"\1", s)
        # end

    @staticmethod
    def strip_accents(s):
        nkfd_form = unicodedata.normalize('NFKD', s)
        return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')

    @staticmethod
    def __filtrar_text(text):
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