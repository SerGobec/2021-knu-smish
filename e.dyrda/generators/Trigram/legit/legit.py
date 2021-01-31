from Adversary import Adversary


def legit():
    gen = Adversary()


    k = 0
    with open('text.txt') as f:
        spam = []
        for text in f.readline().replace('\n','').split('.'):
            spam.append(text)
            texts_gen = gen.generate(spam, text_sample_rate=5.0, word_sample_rate=0.5, max_attacks = 15, attacks={'synonym':0.5,'insert_duplicate_characters':0.3,'change_case':0.2})
            spam.remove(text)
            
            for i in range(len(texts_gen)):
                with open('result.txt', 'a') as f:
                    f.write('LEGI'+'\t'+str(texts_gen[i][0])+'\n')
                f.close()
                k += 1

