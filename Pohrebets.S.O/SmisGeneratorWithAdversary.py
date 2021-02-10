from Adversary import Adversary

#generator is not very usefull, because can only swap words and lettert. You
#must generate sentences.
texts = [
    'Deal with bank user customer alala  ',
    'A law on alcohol is siigned'
]

labels = [
    0,
    1
]
gen = Adversary()
spam_messages = [t for i, t in enumerate(texts) if labels[i] == 1]
texts_gen = gen.generate(spam_messages, text_sample_rate=50.0, word_sample_rate=0.9,
                         attacks={'swap_words': 0.1, 'synonym': 5, 'change_case': 0.0, 'letter_to_symbol': 0, 'delete_characters': 0.0, 'swap_letters': 0.0})

for i in texts_gen:
    print("SMIS " + i[0])