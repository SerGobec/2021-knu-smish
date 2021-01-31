from Adversary import adversary
f = open('gen.txt', 'w')
f.close()      #відкрили-закрили файл задля його очищення

gen = adversary.Adversary(verbose=True, output='Output/')

clear_texts_personal = [
    'hello how are you doing? I spent all day watching films',
    'mark, buy me a coffee pls',
    'mom, i will be late. dont close the door',
    'hi, mary, how are you? i wanna see you next week',
    'ben, dont forget to clean your room or i will thow your phone away!',
    'whats up bro? what about meeting at night?',
    'do you free for a date wednesday?',
    'lets go to the amazing sushi place downtown',
    'dude i will find you anyway, dont try to play with me',
    'leave me alone, we will never be together!',
    'i hate you! get out from my house',
    'buy me a pizza, man. i will send you money later',
    'please give me a last chance, it will not happened again',
    'tell ann to turn on her phone pls, its very important!',
    'hello how your father is?',
    'that was amazing pizza, i want to go there next week again!',
    'cevin, where are you? we are worrying.',
    'come on, dont think about it, everything will be ok',
    'he well meet me after the concert mom',
    'mr. jhonson i will be late for 2 hours, excuse me please',
    'send me your familys photos from weekends, i have not seen you for a hundred years',
    'mike, remember what the doctor said and dont drink alcohol',
    'i will not let you go to that party because it is dangerous for you',
    'i am free now so you can call me',
    'i cant wait for meeting you in june',
    'stop calling me when i am working!',
    'if you forget to buy a beer i will kill you!'
]  #легальні повідомлення персональні
links_cl_per=['']
clear_messages_personal = [t for i, t in enumerate(clear_texts_personal)]
clear_texts_personal_gen = gen.generate(clear_messages_personal, links_cl_per, label='legi', text_sample_rate=9, word_sample_rate=0.4,
                         attacks={'synonym': 0.3, 'letter_to_symbol':0.5, 'num_to_word':0.3, 'insert_punctuation':0.03 ,
                                  'delete_characters': 0.3, 'change_case': 0.4,
                                  'swap_words': 0.4, 'swap_letters': 0.3, 'insert_duplicate_characters': 0.3})
print(clear_texts_personal_gen)

clear_texts_calls = [
    'You have 1 missed call from 89468214. Missed call was at 07:53 PM on 11-Sep-2015.',
    'You have missed calls from 0687651265(3 times at 2019-08-14 12:04:18). Your Kyivstar ',
    'Missed Call from: Boss. Ph.No: 88005553535',
    'Missed Call from: Mother for 2 times. Ph.No: 874652321',
    'From: Aunt. I called you 4 times. The last one at 9:34, 20.10.20',
    'You have 2 missed calls from 4568234. The last missed call was at 09:22 AM on 15-May-2021.',
    'From: Mary. I called you at 18:21, 10.01.21',
    'You missed a call from me at 17:14 07 Oct. This is a free Call Alert from Vodafone',
    'MISSED CALL ALERT - 08004681000 Date: 01/8/2014 15:07:50 Caller Number: 456987220(UK London)',
    'You have 3 missed calls(the last one from Jack at 18:04). Call *466# for details',
    'Received 5/21/2014 4:07 PM - You have a new missed call from Julia Miller(201-547-8965)',
    'You have missed calls from 0554896248(5 times at 2017-04-28 18:45:59). Your Lifecell',
    'MISSED CALL ALERT - 08004681000 Date: 31/02/2019 10:59:01 Caller Number: 9854123654(UKR Kyiv)',
    'You have 4 missed calls from 06756428. The last missed call was at 10:05 AM on 30-Apr-2019.'
] # легальні смс дзвінки
links_cl_ca=['']
clear_messages_calls = [t for i, t in enumerate(clear_texts_calls)]
clear_texts_calls_gen = gen.generate(clear_messages_calls, links_cl_ca, label='legi', text_sample_rate=9, word_sample_rate=0.3,
                                     attacks={'synonym': 0.3, 'letter_to_symbol': 0.05, 'num_to_word': 0.05,
                                              'insert_punctuation': 0.03, 'delete_characters': 0.3, 'change_case': 0.3,
                                              'swap_words': 0.4, 'swap_letters': 0.3,
                                              'insert_duplicate_characters': 0.3})
print(clear_texts_calls_gen)


clear_texts_other = [
    'Tarrif megabytes are ending, the next 100mb/10 UAH(up to 5 additional packages until the end of the day). Check: *252#',
    'Your application for connection a new package was rejected because of the lack of funds. You must have more than 90 UAH on your account. Info: *111#',
    '2020.10.19 at 12:27 your account has been replenished by UAH 150.00. Thank you for being with us!',
    'Your SMS password is 4987. Dont give it to anyone! The password is valid for 3 minutes',
    'Payement for the rattif will be charged off tomorrow night. Recommended balance is mote than 160 UAH, check *111#.',
    'Personal offer! Unlimited internet and calls in-network calls, 300 minutes on other mobiles and abroad calls. Order the tarrif until 10/10/2020 and get a discount of 150 UAH for the first payment!',
    'The Happy Day action has been activated. Unlimite internet and calls in the network will be valid for next 24 hours. Restart your phone to use this promotion.',
    'You are credited with 30 bonuses(1 bonus = 1 UAH) Bonuses can be used for 30 days for paying for calls, tarrifs, SMS or mobile internet. Theck the balance in the application.',
    'Iam online. Call me, please. Number: 789652145',
    'I have already finished the conversation. Call me, please. Number: 12459870',
    'Code for charging off 240 UAH: 40-41-64. Silpo From card 517581******7892',
    'Mike, your tickets are booked. Please, take them from your local railway station',
    'Your package NO1789546 is it your local post branch office.',
    'Application 546656565 is comfired automatically. Date of dispatch is 18.12.2020.',
    'You have open access to the kyivstar TV Easy package. These are 90 free TV channels.',
    'Your balance is UAH 30.25. Recommended sum to pay is UAH 100. Personal ID 6598712',
    'Your registration code: 1595',
    '<#> WhatsApp code: 456-98. Do not share the code with anyone',
    'How are you satisfied with Kyivstar mobile services? Answer with a number from 0 to 10.',
    'Use 60% off discount for next 2 drives with promo code JAN60X2UA.',
    'You have entered the wrong password from your WEB-banking.'
] # легальні смс інші типи
links_cl_ot=['']
clear_messages_other = [t for i, t in enumerate(clear_texts_other)]
clear_texts_other_gen = gen.generate(clear_messages_other,links_cl_ot, label='legi', text_sample_rate=9, word_sample_rate=0.3,
                                     attacks={'synonym': 0.95, 'letter_to_symbol': 0.2, 'num_to_word': 0.1,
                                              'insert_punctuation': 0.03, 'delete_characters': 0.2, 'change_case': 0.4,
                                              'swap_words': 0.4, 'swap_letters': 0.3,
                                              'insert_duplicate_characters': 0.3})
print(clear_texts_other_gen)


def smish_message(text, links):    # функція генерації смішингових смс
    smish_messages = [t for i, t in enumerate(text)]
    smish_texts_gen = gen.generate(smish_messages, links, label='smis', text_sample_rate=5, word_sample_rate=0.5,
                                   attacks={'synonym': 0.7, 'letter_to_symbol': 0.6, 'num_to_word': 0.3,
                                            'insert_punctuation': 0.07, 'delete_characters': 0.3, 'change_case': 0.5,
                                            'swap_words': 0.3, 'swap_letters': 0.4, 'insert_duplicate_characters': 0.3})
    print(smish_texts_gen)


smish_texts_type1 = [
    'FRM:MONOBANK SUBJ: MONOBANK Unusual Activity! MSG: Acc Frozen TO UNLOCK GO = ',
    'Your bank account was blocked. Please, follow this link for future instructions: ',
    'FROM: PRIVATBANK (ALERT!) Your account is limited. To remove limits visit: ',
    'Your internet banking has been disabled for security reasons, please visit your local branch or unlock at: ',
    'ATTENTION! Your account has been suspected and requires verification. Details: ',
    'All your transactions are suspended. You must confirm your device immediately. Follow: '
]  # смішинг смс 1 тип
links_sm_1=['http://btcpayment.000webhostapp.com/login.html', 'http://baanvitaminsea.com/uploads/uploads/form/login.htm	',
            'https://drcarmenmora.com/http/https/da/index.html', 'https://ekoda-recording.tokyo/wp-includes/oralivechief/oralive/',
            'https://normativaeuropea-aggiornata.com/', 'http://www.myenterpriseworld.com/mail/']*5
smish_texts_type2 = [
    'Dear client, your credit card was blocked. For further information visit local branch or follow this link: ',
    'ALERT! Your card is FROZEN! UNLOCK AT ',
    'FROM: PRIVAT24 SUBJ: CARD ALERT Details at ',
    '(ALERT!) your credit card has been suspected and is deactivated now. Visit this site for activation: ',
    'Dear bank user, we have detected some unusual activity at your credit card. We ask you to follow this link to update your card:  ',
    'TRANSACTION UAH 4652.00 to 5168******7865 at 10:45 PM. To confirm ignore this message, to decline follow instructions: '
]
links_sm_2=['https://www.boiverify.com/Login.php', 'https://www.formedlicensing.com/wp-content/HDK/DHL.13.0.1/source/verify.php?email=', 'https://ee.securing-details.com/login/index.php',
            'http://103.125.189.202/hotvideopro/hekziw543058', 'http://dev.klinikmatanusantara.com/log-au/mpp/signin/8073/websrc',
            'http://forb-fbookcom-854101193154.ctiaspire.com/gate.html?location=913b58aa869d4d4fe7e6b244843d02b7']*5

smish_texts_type3 = [
    'Great news! You won $1000! Visit our site for getting your prize',
    'Alert! You was selected as a winner of $150000!!! Enter your personal code to get money at ',
    'Congratulations Mark, you have been won our mystery box this week - please use this to shedule a delivery: ',
    'Congrats! You were chosen as a super-prize winner! Log here to get your present: ',
    'Dear Ann! Today Adidas store announced there lottery winners and you took 3rd place. Details here: ',
    'You have 12 hours to pick your $1000 gift up. For details call 078654125',
    'Get your FREE IPHONE 11 PRO MAX from nearest store! Call at 056478922 and tell your personal ID85412'
]
links_sm_3=['https://dhlvideo.id/login.../', 'http://vieuxshack.com/download/adobe/ceaa59e4265079f481f9e59127697b5b/login.php?cmd=login_submit&id=4d6554b36c83db47c52f77d7c1a0842a4d6554b36c83db47c52f77d7c1a0842a&session=4d6554b36c83db47c52f77d7c1a0842a4d6554b36c83db47c52f77d7c1a0842a',
            'http://cla2020gov.com/step2.php', 'http://rmv-322754737.adventistgh.org/gate.html', 'https://www.paypal-merchant.ru/micrositeportal/',
            '', '']*5

smish_texts_type4 = [
    'Must we declare war to Gonduras? Help our country and vote at an online referendum! Personal gifts from president for first 1000 voters: ',
    'Nova poshta decided to improve its service and gives free $10 just for taking part in short survey: ',
    'Dont waste your time on working or studying! We opened new service where you can get $7-49 just for watching short videos. Welcome: ',
    'You received 2 bitcoins from anonim user. Register now to accept this transaction: ',
    'You were a good boy and OLX have a present for you! Click here to get it: ',
    'Someone wants to send you $1675.49! You have 10 minutes to accept money - '
]
links_sm_4=['https://new-3-login.com/log.php', 'http://rickstv.com/pop/smtp/authenticate.php?INFO=unsuccessful&email=',
            'http://hurtfulwellofffinance--five-nine.repl.co/WACTH-VIDEO-195.114.145.90#0.3074481465350807',
            'https://data.cloudsave247.com/', 'http://misha-now.today/view-signin.php?facebook_com/marketplace/item/691188050=',
            'http://rickstv.com/pop/smtp/authenticate.php?INFO=unsuccessful&email=']*5

smish_texts_type5 = [
    'Your YouTube channel was blocked for an unusual activity. You have to validate it here: ',
    'Google has frozen your account due to lack of personal information. Solve it following this link: ',
    'Your ITunes account has been locked. Find details here: ',
    'Your AppleID has been suspended because we are unable to verify your information. To unlock it go here: ',
    '(GMAIL) Your account is limited. Please follow this link to update your personal infrmation: ',
    'FROM: GOOGLE. your PASSWORD was CHANGED. Restore it here if it was no you: ',
    'It seems to us that your Yahoo account was stolen. Call to 087456321 to talk with our operator'
]
links_sm_5=['https://staging.participatorybudgeting.org/wp-admin/user/net-acc/supervisor.cnfg.ld-details.net/Nz-log/Nz/Webme/Log/get_started/',
            'https://greenhues.co.in/', 'https://lamoorespizza.com/Dropbox/dropbox/o1/main.html', 'http://dev.klinikmatanusantara.com/log-au/mpp/signin/4fd3/websrc',
            'http://sjdhomes.co.uk/wp-content/themes/excel-rd42/?_sm_au_=irmj6vs0qpkqb50qbqvgvk7jj80tt', 'https://mail.igvc.link/xxxxxx',
            '']*5

spam_texts = [
    'Denis, you have free $1 promo code. Us it here: ',
    '0.01% loan was approved for you! Get it: ',
    'Buy new shoes with great discount: ',
    'Only today and only fo you - 40% discount on our phones. Buy new one: ',
    'Dear customer, choosing your favorite channels just became as easy as 1-2-3. Try it now: ',
    'You dont have GoodCashApp money account. Please visit this link and download our app: ',
    'GagaCell: You are not subscribed to any services on 28765. For more info go to: ',
    'We give money for everything! Loan up to UAH 5000000. Discounts on target purchases. INFO: ',
    'BENEFIT! Get UAH 12000 for 0 percents! Get money: ',
    '1+1 the second pizza free for 24 hours. order a delivery: '
]  # спам смс
links_sp=['https://aliexpress.ru/', 'https://creditkasa.ua/', 'https://www.fotshop.se/', 'https://allo.ua/ru/',
          'https://kyivstar.ua/uk/home-kyivstar/kyiv?banner=main_slider', 'https://goodcash.co/', 'https://play.google.com/store/apps/details?id=krowel.apps&hl=uk&gl=US',
          'https://loany.com.ua/', 'https://moneyveo.com.ua/uk/main/', 'https://mistercat.com.ua/']*8

smish_message(smish_texts_type1, links_sm_1)
smish_message(smish_texts_type2, links_sm_2)
smish_message(smish_texts_type3, links_sm_3)
smish_message(smish_texts_type4, links_sm_4)
smish_message(smish_texts_type5, links_sm_5)


spam_messages = [t for i, t in enumerate(spam_texts)]
spam_texts_gen = gen.generate(spam_messages, links_sp, label='spam', text_sample_rate=8, word_sample_rate=0.5,
                              attacks={'synonym': 0.7, 'letter_to_symbol': 0.6, 'num_to_word': 0.3,
                                       'insert_punctuation': 0.07, 'delete_characters': 0.3, 'change_case': 0.5,
                                       'swap_words': 0.3, 'swap_letters': 0.4, 'insert_duplicate_characters': 0.3})
print(spam_texts_gen)