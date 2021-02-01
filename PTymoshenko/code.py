import pandas as pd 
from Adversary import Adversary
smishing0 = []
# text from images from twitter, pinterest , google images etc
with open("smishing.txt") as smishf:
    for i in smishf.readlines():
        smishing0.append(i)

gen = Adversary()
smish = [i[0] for i in gen.generate(smishing0, text_sample_rate=2, word_sample_rate=1,
                         attacks={'synonym': 2} )]

dataset1 = pd.read_csv("SMSSpamCollection.csv", header=0, names=['label', 'sms']) # UCI’s Machine Learning Repository
ds = dataset1.drop(columns="label")
ds["label"] = dataset1["label"].apply(lambda x:"SPAM" if 'spam' in str(x) else "LEGI")

legi = list(ds.loc[ds["label"]=="LEGI"][:1050]["sms"])
spam = list(ds.loc[ds["label"]=="SPAM"][:190]["sms"])

with open("databasePT.txt", "w") as dbtext:
    for label,lpart in (("LEGI",legi),("SPAM",spam), ("SMIS",smish)):
        for mes in lpart:
            print(label+" "+mes, file = dbtext)





print("Nu of sms :: " ,len(legi)+len(spam)+len(smish))
#
#
# And the final step included straight work with dataset 