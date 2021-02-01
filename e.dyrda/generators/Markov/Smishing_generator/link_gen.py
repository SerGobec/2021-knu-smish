import random

def count_lines(file):
    try:
        with open(file) as fl:
            return len(fl.readlines())
    except Exception:
    
        return 0
        
        
def gen_char(minrange,maxrange,table):
    x = random.randrange(0,maxrange[(count_lines("gen.txt"))-1],1)
    for q in range (count_lines("gen.txt")):
        if (x>=minrange[q])and(x<=maxrange[q]):
            return table[q]
            
def chck_buk(ty,nu,bo1,bo2,bo3):
    if ((ty[nu-2] == bo1) and (ty[nu-1] == bo2) and (ty[nu] == bo3)):
        return True
    else:

        return False
        
        
def chck_chr(ch,nu,snu,sch):

    if snu == nu:
        if (ch == sch):
            return True
        else:
            return False
    else:
        return False
        
def chck_ap(ty,nu):
    if (ty[nu-1] == 2) and (ty[nu] == 2):
        return True
    else:
        if (ty[nu-1] == 1) and (ty[nu] == 2):
            return True
        else:
            return False






                

