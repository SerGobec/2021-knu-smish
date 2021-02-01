import io
lines=0
with io.open('open_db.txt', encoding='utf-8') as f:
    for line in f:
        lines = lines + 1

print(lines)

leg=0
sp=0
sm=0
with io.open('open_db.txt', encoding='utf-8') as f:
    for line in f:
        if "LEGI" in line:
            leg+=1
        if "SPAM" in line:
            sp+=1
        if "SMIS" in line:
            sm+=1

print("LEGI:", leg)
print("SPAM:", sp)
print("SMIS:", sm)
