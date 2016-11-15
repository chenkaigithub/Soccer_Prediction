import csv

years = ['05','06','07','08','09','10','11','12','13','14','15','16']
fieldnames = ['Year','HomeTeam','AwayTeam','Diff']

def handleFile(filePath,year):
	with open(filePath) as csvfileread:
		reader = csv.DictReader(csvfileread)			
		for row in reader:	
			if(row['HomeTeam'] == '' or row['AwayTeam'] == ''):
				continue
			row = {'Year':year,'HomeTeam':row['HomeTeam'],'AwayTeam':row['AwayTeam'],'Diff':(int(row['FTHG'])-int(row['FTAG']))}
			writer.writerow(row)
			
def handleChampionshipLeague():
	for year in years:
		handleFile('./English_League_Games/Championship/CH_'+year+'.csv',year)
		
def handleLeagueOne():
	for year in years:
		handleFile('./English_League_Games/L1/L1_'+year+'.csv',year)
		
def handleLeagueTwo():
	for year in years:
		handleFile('./English_League_Games/L2/L2_'+year+'.csv',year)
			
def handlePremierLeague():
	for year in years:
		handleFile('./English_League_Games/Premier/PR_'+year+'.csv',year)

with open('output.csv', 'w', newline='') as csvfilewrite:
	writer = csv.DictWriter(csvfilewrite, fieldnames=fieldnames)
	writer.writeheader()
	handlePremierLeague()
	handleChampionshipLeague()
	handleLeagueOne()
	handleLeagueTwo()
		



