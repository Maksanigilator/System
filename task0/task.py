import csv

with open('task2.csv') as f:
    #reader = csv.reader(f)
    reader = csv.reader(f, delimiter=':', quoting=csv.QUOTE_NONE)
    for row in reader:
        print(row)

