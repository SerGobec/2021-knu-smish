from Adversary import Adversary
f = open('text.txt', 'w', encoding="utf-8")

texts = [ "messages"]

gen = Adversary()
texts_gen = gen.generate(texts, text_sample_rate=5.0, word_sample_rate=0.5,
                         attacks={'synonym': 2, 'letter_to_symbol': 0.3})
for i in range(0, 190):
    f.write("SMIS - " + str(texts_gen[i][0]) + "\n") #here were "LEGI -", "SPAM -" and "SMIS - " and i was changing number of messages depand on messages in texts

f.close()