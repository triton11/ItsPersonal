import csv

meyersToNumbersDict = {
    "ISTJ": 0,
    "INTP": 1,
    "ISFJ": 2,
    "INFJ": 3,
    "ISTP": 4,
    "ISFP": 5,
    "INFP": 6,
    "INTJ": 7,
    "ESTP": 8,
    "ESTJ": 9,
    "ESFJ": 10,
    "ENFJ": 11,
    "ESFP": 12,
    "ENTJ": 13,
    "ENTP": 14,
    "ENFP": 15
}

counts = {
    0: "ISTJ",
    1: "INTP",
    2: "ISFJ",
    3: "INFJ",
    4: "ISTP",
    5: "ISFP",
    6: "INFP",
    7: "INTJ",
    8: "ESTP",
    9: "ESTJ",
    10: "ESFJ",
    11: "ENFJ",
    12: "ESFP",
    13: "ENTJ",
    14: "ENTP",
    15: "ENFP"
}
sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#1735
with open('data/test_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            sums[int(row[0])] += 1
        line_count += 1
for i in range(0, len(sums)):
    sums[i] = (sums[i]/1767.0) * 100
#print(sums)

sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#1735
with open('data/train_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            sums[int(row[0])] += 1
        line_count += 1

for i in range(0, len(sums)):
    sums[i] = (sums[i]/7000.0) * 100
for i in range(len(sums)):
    print(sums[i], counts[i])