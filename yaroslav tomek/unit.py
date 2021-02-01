import io
import os

with io.open('open_db.txt', encoding='utf-8') as f_in:
    text=f_in.read()
with io.open('gen.txt', encoding='utf-8') as f_in2:
    text1=f_in2.read()
print(text)
print(text1)
os.chdir("data")
with io.open('dataset.txt', 'w', encoding='utf-8') as f_out:
    f_out.write(text1)
    f_out.write(text)