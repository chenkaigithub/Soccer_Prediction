import csv
with open('./English_League_Games/Premier/PR_16.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['Date'])
