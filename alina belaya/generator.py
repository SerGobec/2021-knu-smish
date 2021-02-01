"""
SMS generator from https://github.com/airbnb/artificial-adversary
SMS dataset from https://www.kaggle.com/uciml/sms-spam-collection-dataset and https://www.pinterest.com.au/seceduau/smishing-dataset/
"""


from Adversary import Adversary

f = open('sms_dataset.txt', 'w', encoding="utf-8")

texts = [
    "YOU ARE CHOSEN TO RECEIVE A £350 AWARD! Pls call claim number 09066364311 to collect your award which you are selected to receive as a valued mobile customer.",
    "Congratulations, Your entry into Our contest last month made you a WINNER! Goto http://www.apple.com.textwon.com to claim your prize! You have 24 hours to claim",
    "Your mobile Number has WON , 615,000 Million Pounds in Apple iPhone UK. Ref No: NK115G. For claim Email your name, Country & Occupation to:freeappleiphone@w.cn",
    "Missed call alert. These numbers called but left no message. 07008009200",
    "You have been selected to stay in 1 of 250 top British hotels - FOR NOTHING! Holiday Worth £350! To Claim, Call London 02072069400. Bx 526, SW73SS",
    "Hi, this is Mandy Sullivan calling from HOTMIX FM...you are chosen to receive £5000.00 in our Easter Prize draw.....Please telephone 09041940223 to claim before 29/03/05 or your prize will be transferred to someone else....",
    "UR GOING 2 BAHAMAS! CallFREEFONE 08081560665 and speak to a live operator to claim either Bahamas cruise of£2000 CASH 18+only. To opt out txt X to 07786200117",
    "Todays Voda numbers ending with 7634 are selected to receive a £350 reward. If you have a match please call 08712300220 quoting claim code 7684 standard rates apply.",
    "Call FREEPHONE 0800 542 0578 now!",
    "URGENT, IMPORTANT INFORMATION FOR O2 USER. TODAY IS YOUR LUCKY DAY! 2 FIND OUT WHY LOG ONTO HTTP://WWW.URAWINNER.COM THERE IS A FANTASTIC SURPRISE AWAITING FOR YOU",
    "You have 1 new voicemail. Please call 08719181503",
    "network operator. The service is free. For T & C's visit 80488.biz",
    "sexy sexy cum and text me im wet and warm and ready for some porn! u up for some fun? THIS MSG IS FREE RECD MSGS 150P INC VAT 2 CANCEL TEXT STOP",
    "Our brand new mobile music service is now live. The free music player will arrive shortly. Just install on your phone to browse content from the top artists.",
    "22 days to kick off! For Euro2004 U will be kept up to date with the latest news and results daily. To be removed send GET TXT STOP to 83222",
    "Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+",
    "Ever thought about living a good life with a perfect partner? Just txt back NAME and AGE to join the mobile community. (100p/SMS)",
    "Talk sexy!! Make new friends or fall in love in the worlds most discreet text dating service. Just text VIP to 83110 and see who you could meet.",
    "U have a secret admirer who is looking 2 make contact with U-find out who they R*reveal who thinks UR so special-call on 09058094565"
    ]

labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
]
gen = Adversary()

spam_messages = [t for i, t in enumerate(texts) if labels[i] == 0]
spam_gen = gen.generate(spam_messages, text_sample_rate=5.0, word_sample_rate=0.8,
                         attacks={'synonym': 0.5, 'change_case': 0.5, 'letter_to_symbol': 0.5, 'delete_characters': 0.2})
for i in spam_gen:
    print('SPAM  ', i[0])

smis_messages = [t for i, t in enumerate(texts) if labels[i] == 1]
smis_gen = gen.generate(smis_messages, text_sample_rate=5.0, word_sample_rate=0.8,
                         attacks={'synonym': 0.5, 'change_case': 0.5, 'letter_to_symbol': 0.5, 'delete_characters': 0.2})
for i in smis_gen:
    print('SMIS  ', i[0])

