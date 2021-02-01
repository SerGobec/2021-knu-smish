from link_gen import *
import random
from Adversary import Adversary

def link_generation():
    
    o = []

    inc = 0
    sup = 0
    ot = 0
    legit_domens = ['google.com',
                    'apple.com',
                    'www.paypal.com'
        ]
    n = random.randint(1,10)
    k = random.randint(0, len(legit_domens)-1)
    domen = legit_domens[k]
    if n <= 2:
        for i in range(100):
            with open("gen.txt") as f:
                char=[]
                ver=[]
                i=0
                ran1=[0]
                ran2=[]
                q=0
                e=0
                name=""
                nn = 0
                d = 7
                type = []

                
                for i in range (count_lines("gen.txt")):
                    s = f.readline()
                    n = s[0:1]
                    m = s[2:5]
                    char.append(n)
                    ver.append(m)
                   
                    if i == 0:
                        ran2.append(int(ver[i]))
                    else:
                        ran1.append(ran2[i-1]+1)
                        ran2.append(int(ran1[i]+int(ver[i])))
                         
                       
                while q < d:
                    p = gen_char(ran1,ran2,char)
                    
                    if (p in 'aeiouy'):
                        type.append(1)
                    else:
                        if (p == '`'):
                            type.append(2)
                        else:
                            type.append(0)

                            
                        
                    if q<2:
                        if (q==0)and(p == '`'):
                            type.pop(q)
                        else:            
                            name = name + p
                            q = q + 1
                    else:
                        if ((chck_buk(type,q,1,1,1)) or (chck_buk(type,q,0,0,0)) or (chck_buk(type,q,0,2,1)) or (chck_buk(type,q,0,0,0))):
                            type.pop(q)
                        else:
                            if chck_chr(p,q,d-1,'`'):
                                type.pop(q)
                            else:
                                if chck_ap(type,q):
                                    type.pop(q)
                                else:
                                    name = name + p
                                    q = q + 1

            
            o.append('http://'+ domen + ' ' + 'http://bit.ly/'+ name + '/')
            
                    
            for i in range(len(o)):
                with open("data.txt", 'a') as h:
                    h.write(str(o[i])+'\n')
            
    else:
        gen = Adversary()
        fake_domen = gen.generate(domen.split(), text_sample_rate=8.0, word_sample_rate=0.5, max_attacks = 2, attacks={'num_to_word': 1.0, 'letter_to_symbol': 1.0})
        for i in range(0,len(fake_domen)-1):
            if str(fake_domen[i][0]) == str(domen):
                pass
            else:
                o.append('http://'+ domen + ' ' + 'http://'+str(fake_domen[i][0]))
                break
        for i in range(len(o)):
                with open("data.txt", 'a') as h:
                    h.write(str(o[i])+'\n')

           

    

        
