#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
from random import uniform
from collections import defaultdict

from smishing import *

r_alphabet = re.compile(u'[а-яА-Яa-zA-Z0-9]+|[.,:;?!+]+')

def gen_lines(corpus):
    data = open(corpus, encoding="utf-8")
    for line in data:
        yield line.lower()

def gen_tokens(lines):
    for line in lines:
        for token in r_alphabet.findall(line):
            yield token

def gen_trigrams(tokens):
    t0, t1 = '$', '$'
    for t2 in tokens:
        yield t0, t1, t2
        if t2 in '.?':
            yield t1, t2, '$'
            yield t2, '$','$'
            t0, t1 = '$', '$'
        else:
            t0, t1 = t1, t2

def train(corpus):
    lines = gen_lines(corpus)
    tokens = gen_tokens(lines)
    trigrams = gen_trigrams(tokens)

    bi, tri = defaultdict(lambda: 0.0), defaultdict(lambda: 0.0)

    for t0, t1, t2 in trigrams:
        bi[t0, t1] += 1
        tri[t0, t1, t2] += 1

    model = {}
    for (t0, t1, t2), freq in tri.items():
        if (t0, t1) in model:
            model[t0, t1].append((t2, freq/bi[t0, t1]))
        else:
            model[t0, t1] = [(t2, freq/bi[t0, t1])]
    return model

def generate_sentence(model):
    phrase = ''
    t0, t1 = '$', '$'
    while 1:
        t0, t1 = t1, unirand(model[t0, t1])
        if t1 == '$':
            break
        if t1 in ('.!?,;:') or t0 == '$':
            phrase += t1
        else:
            phrase += ' ' + t1
    return phrase.capitalize()

def unirand(seq):
    sum_, freq_ = 0, 0
    for item, freq in seq:
        sum_ += freq
    rnd = uniform(0, sum_)
    for token, freq in seq:
        freq_ += freq
        if rnd < freq_:
            return token

if __name__ == '__main__':
    files = ['smishing_youhavewon.txt',
             'smishing_fakesupport.txt',
             'smishing_paytowin.txt',
             'smishing_fakelogin.txt',
             'smishing_youwillwinforreal.txt']
    cc = []
    for i in range(len(files)):
        k = 0
        file = files[i]
        while k != 4:
            model = train(file)
            mm = open(file).readlines()
            a = generate_sentence(model)
            if a not in mm: 
                if a not in cc:
                    if len(a.split()) >= 4:
                        cc.append(a)
                        k += 1
                    else:
                        pass
                else:
                    pass
            else: pass
    with open('text.txt', 'w') as f:
        for i in cc:
            f.write(i)
    f.close()
    smishing()

        
