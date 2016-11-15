import csv

fieldnames = ['Year','HomeTeam','AwayTeam','Diff']

index = 0
dict = {}

with open('team_indexes.csv', 'w', newline='') as csvfileindexes:
	with open('games.csv', 'w', newline='') as csvfilewrite:
		writer = csv.DictWriter(csvfilewrite, fieldnames=fieldnames)
		writer.writeheader()
		writerIndexes = csv.DictWriter(csvfileindexes, fieldnames=['Name','Index'])
		writerIndexes.writeheader()
		with open('output.csv') as csvfileread:
			reader = csv.DictReader(csvfileread)			
			for row in reader:
				if(not dict.__contains__(row['HomeTeam'])):
					dict[row['HomeTeam']] = index;
					writerIndexes.writerow({'Name':row['HomeTeam'],'Index':index})
					index = index + 1
				if(not dict.__contains__(row['AwayTeam'])):
					dict[row['AwayTeam']] = index;
					writerIndexes.writerow({'Name':row['AwayTeam'],'Index':index})
					index = index + 1
				temp_row = {}
				temp_row['Year'] = row['Year']
				temp_row['HomeTeam'] = dict[row['HomeTeam']]
				temp_row['AwayTeam'] = dict[row['AwayTeam']]
				temp_row['Diff'] = row['Diff']
				writer.writerow(temp_row)