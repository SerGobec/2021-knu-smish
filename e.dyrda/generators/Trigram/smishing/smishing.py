from Adversary import Adversary

def smishing():
    gen = Adversary()


    k = 0
    with open('text.txt') as f:
        spam = []
        for text in f.readline().replace('\n','').split('.'):
            spam.append(text)
            texts_gen = gen.generate(spam, text_sample_rate=10.0, word_sample_rate=0.5, max_attacks = 15, attacks={'synonym': 0.5, 'num_to_word': 0.5, 'letter_to_symbol': 0.5, 'change_case': 0.5})
            spam.remove(text)
            for i in range(len(texts_gen)):
                with open('smis.txt', 'a') as f:
                    f.write('SMIS'+'\t'+str(texts_gen[i][0])+'\n')
                f.close()
                k += 1
    print(k)
