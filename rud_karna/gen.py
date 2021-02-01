"""

DB sources:
SMS Spam Collection from Data World https://data.world/lylepratt/sms-spam
and one json file with australian sms (lost link),
screenshots from big amount of websites 
including Twitter and Pinterest (nice smishing collection by Jeremy Lee https://co.pinterest.com/seceduau/smishing-dataset/)
https://openphish.com - for smish links
and Adversary

"""



from Adversary import Adversary

texts = ['how is fomic acid prepared in a lab. give chemical reaction ',
         'Someone who came in contact with you tested positive or has shown synptoms for COVID-19 & recommends you self-isolate/get tested. More at',
         'Due to the recent shortage of face masks, the Red-Cross will be distributing one free box per household. Visit  to get one.']

labels = [0, 1, 2]

gen = Adversary()
messages = [t for i, t in enumerate(texts)]
text_gen = gen.generate(messages,
                        text_sample_rate=6.0,
                        word_sample_rate=0.5,
                        attacks={'synonym': 0.5,
                                 'change_case': 0.8,
                                 'letter_to_symbol': 0.5,
                                 'delete_characters': 0.3})

for j in range(0, len(text_gen)):
    print(text_gen[j][0])


"""
For counting size of our DB
"""
import pandas as pd
df = pd.read_table('smsdb.txt', sep='\n', delimiter='\t',  header=None)

print(len(df))

print(df[(df[0] == "LEGI")].count())
print(df[(df[0] == "SMIS")].count())
print(df[(df[0] == "SPAM")].count())
#print(df)
"""
LEGI: 70- 1050
SMIS: 20- 300
SPAM: 10- 150

"""