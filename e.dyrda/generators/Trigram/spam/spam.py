from Adversary import Adversary


def spam():
    gen = Adversary()


    k = 0
    with open('tekst.txt') as f:
        spam = []
        for text in f.readline().replace('\n','').split('.'):
            spam.append(text)
            texts_gen = gen.generate(spam, text_sample_rate=7.0, word_sample_rate=5.0, max_attacks = 15, attacks={'delete_characters':0.6,'swap_words':0.7,'synonym':0.4,'swap_letters':0.6,'insert_duplicate_characters':0.8})
            spam.remove(text)
            
            for i in range(len(texts_gen)):
                with open('res.txt', 'a') as f:
                    f.write('SPAM'+'\t'+str(texts_gen[i][0])+'\n')
                f.close()
                k += 1

