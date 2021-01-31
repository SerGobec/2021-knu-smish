from Dictogram import Dictogram
import random
from collections import deque
import re
from markov_model import make_markov_model
from link_generation import *



def generate_random_start(model):
    if 'END' in model:
        seed_word = 'END'
        while seed_word == 'END':
            seed_word = model['END'].return_weighted_random_word()
        return seed_word
    k = list(model.keys())
    return random.choice(k)


def generate_random_sentence(length, markov_model):
    current_word = generate_random_start(markov_model)
    sentence = [current_word]
    i = 0
    while i <= int(length-1):
        if current_word in list(markov_model.keys()):
            current_dictogram = markov_model[current_word]
            current_word = current_dictogram.return_weighted_random_word()
            sentence.append(current_word)
            i += 1
        else: 
            sentence[0] = sentence[0].capitalize()
            return ' '.join(sentence) + '.'
    sentence[0] = sentence[0].capitalize()
    return ' '.join(sentence) + '.'


files = ['smishing_youhavewon.txt',
             'smishing_fakesupport.txt',
             'smishing_paytowin.txt',
             'smishing_fakelogin.txt',
             'smishing_youwillwinforreal.txt']

links = []
link_generation()
with open('data.txt') as file:
    links = file.readlines()

for m in range(len(files)):
    file = files[m]
    with open(file) as f:
        b = str(f.readlines()).lower()
        b = b.replace("['",' ').replace("']",' ').replace("\n",' ').replace('.',' ').split()
    



    a = make_markov_model(b)

    l = []
    c = 0
    k = 0

    while c <= 40 and k <= 100:
        d = []
        d = generate_random_sentence(11, a)
        if str(d) not in l and len(str(d).split()) >= 5:
            l.append(str(d)+' ')
            c += 1
        else: 
            k += 1
    
    

    with open('text.txt', 'a') as f:
        for i in l:
            n = random.randint(0,len(links))
            link = links[n-1]
            position = random.randint(0,len(i))
            i = i[:position]+' '+link+' '+i[position:]            
            f.write('SMIS'+'\t'+i.replace("\n",'').replace("\n.",'.')+'\n')
    f.close()
