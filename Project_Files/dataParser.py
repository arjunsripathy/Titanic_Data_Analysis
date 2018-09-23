import numpy as np
import csv

rawD = []
with open("data.csv",'r') as csvfile:
		reader = csv.reader(csvfile,dialect=csv.excel_tab)
		for row in reader:
				rawD.append(row[0].split(','))
rawD = rawD[1:]


''' Raw Data
	0 -> pass no. not used.
	1 -> survival already numerical
	2 -> class already numerical
	3/4 -> name not used
	5 -> male:0, female:1
	6 -> Age already numerical
	7 -> #sib/sp already numerical
	8 -> #par/kids already numerical
	9 -> ticket number not used
	10 -> fare already numerical
	11 -> cabin not used
	12 -> start Loc. C/Q/S (Sep into 3 vars)

	Atts
	0->class
	1->gender
	2->#sib/sp
	3->#par/kids
	4->fare
	5->C
	6->Q
	7->S
	8->Age
	9->title
	10->SURVIVAL
'''

titleZeros = ['Mr','Mrs','Miss','Ms','Mme','Mlle','Sir','Lady']
titleOnes = ['Master','Don','Rev','Dr','Major','the Countess','Jonkheer','Col','Capt']
zD = zip(titleZeros,[0]*len(titleZeros))
oD = zip(titleOnes,[1]*len(titleOnes))
titleDict = dict(zD+oD)

def processData(raw):

	numAtts = 10

	d1 = {'male':0,'female':1}
	d5 = {'C':1,'Q':0,'S':0}
	d6 = {'C':0,'Q':1,'S':0}
	d7 = {'C':0,'Q':0,'S':1}

	data = []

	d5Mean = 0.189
	d6Mean = 0.087
	d7Mean = 0.724
	d8Mean = 29.7

	for i in range(len(raw)):
		rawRow = raw[i]
		dRow = np.zeros(numAtts+1)
		if(rawRow[2]=='\"Kelly'):
			print(i)
		dRow[0] = int(rawRow[2])
		dRow[1] = d1[rawRow[5]]
		dRow[2] = int(rawRow[7])
		dRow[3] = int(rawRow[8])
		dRow[4] = float(rawRow[10])
		if(rawRow[12]!=''):
			dRow[5] = d5[rawRow[12]]
			dRow[6] = d6[rawRow[12]]
			dRow[7] = d7[rawRow[12]]
		else:
			dRow[5] = d5Mean
			dRow[6] = d6Mean
			dRow[7] = d7Mean
		if(rawRow[6]!=''):dRow[8] = float(rawRow[6])
		else:dRow[8] = d8Mean

		name = rawRow[4][1:]
		title = name[:name.index('.')]
		dRow[9] = titleDict[title]

		dRow[10] = int(rawRow[1])
		data.append(dRow)

	return np.array(data)



np.savetxt("data.txt",processData(rawD))




