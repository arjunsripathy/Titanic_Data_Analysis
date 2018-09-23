import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model

d = pd.read_csv("data.csv")
'''
rows = d.shape[0]
cols = d.shape[1]
g = sns.FacetGrid(d,row="Embarked",col="Survived",height=2,aspect=2)
g.map(sns.barplot,"Sex","Fare")
plt.show()'''
#print(d.shape)
d = d.drop(['Ticket','Cabin'],axis=1)
#print(d.shape)

d['Title'] = d['Name'].str.extract(r'([A-Za-z]+)\.')
#print(d.shape)
#print(d['Title'].head(10))
d["Title"] = d["Title"].replace(to_replace=['Capt','Col','Countess','Don','Dr',
	'Jonkheer','Lady','Major',"Rev","Sir"],value='Rare')
d["Title"] = d["Title"].replace(to_replace={'Mlle':'Miss',"Mme":'Mrs',"Ms":"Miss"})
#ct = pd.crosstab(d['Title'],d['Sex'])
#print(ct)
#sr = d[["Survived","Title"]].groupby(["Title"]).mean()
#print(sr)
#print(d.head(10))
m = {'Mr':1,'Rare':0,'Master':0,'Miss':0,'Mrs':0}
d["Mr"] = d["Title"].map(m)
m = {'Mr':0,'Rare':1,'Master':0,'Miss':0,'Mrs':0}
d["Rare"] = d["Title"].map(m)
m = {'Mr':0,'Rare':0,'Master':1,'Miss':0,'Mrs':0}
d["Master"] = d["Title"].map(m)
m = {'Mr':0,'Rare':0,'Master':0,'Miss':1,'Mrs':0}
d["Miss"] = d["Title"].map(m)
m = {'Mr':0,'Rare':0,'Master':0,'Miss':0,'Mrs':1}
d["Mrs"] = d["Title"].map(m)
d[["Mr","Rare","Master","Miss","Mrs"]] = d[["Mr","Rare","Master","Miss","Mrs"]].astype('int64')
#print(d.head(10))
d = d.drop(labels=["Name","PassengerId"],axis=1)
#print(d.shape)
#print(d.shape)
#print(d.columns)
m = {"male":0,"female":1}
d["Sex"] = d["Sex"].map(m)
#print(d.dtypes)
m = {'C':1,'Q':0,'S':0}
d["C"] = d["Embarked"].map(m)
m = {'C':0,'Q':1,'S':0}
d["Q"] = d["Embarked"].map(m)
m = {'C':0,'Q':0,'S':1}
d["S"] = d["Embarked"].map(m)
print(d[["C","Q","S"]].mean())
d[["C","Q","S"]] = d[["C","Q","S"]].fillna(0)
print(d[["C","Q","S"]].mean())
d = d.drop("Embarked",axis=1)
#g = sns.FacetGrid(d,row="Pclass",col="Title",height=2)
#g.map(plt.hist,"Age")
#plt.show()
#print(d[["Age"]].info())
#act = np.zeros([3,5])
for pc in range(1,4):
	for t in range(1,6):
		aMed = d[(d["Title"]==t) & (d["Pclass"]==pc)]["Age"].dropna().median()
		d.loc[(d["Age"].isna()) & (d["Pclass"]==pc) & (d["Title"]==t),"Age"]= aMed
		#act[pc-1][t-1] = aMed

d = d.drop("Title",axis=1)

#d["Age"] = pd.cut(d["Age"],bins=[0,5,15,30,50,65,d["Age"].max()],labels=False)
#means = d.groupby(["Age"])["Survived"].mean()


ats = [0,5,15,25,35,55,d["Age"].max()]
for i in range(len(ats)-1): d.loc[(d["Age"]>ats[i]) & (d["Age"]<=ats[i+1]),"Age"]=i
d["Age"] = d["Age"].astype("int64")
#print(d[["Age","Survived"]].groupby("Age").mean())

d["FamilySize"] = d["SibSp"]+d["Parch"]+1
#print(d.shape)
#sf = d[["FamilySize","Survived"]].groupby("FamilySize")
#print(sf.count())
#print(sf.mean())
d["isAlone"] = 0
#rint(d.shape)
d.loc[d["FamilySize"]==1,"isAlone"]=1
#s = "Parch"
#print(d[[s,"Survived"]].groupby(s).mean())
d = d.drop(labels=["FamilySize","SibSp","Parch"],axis=1)
#print(d.shape)

#mostFreqPort = d["Embarked"].dropna().mode()[0]
#print(mostFreqPort)
#d["Embarked"] = d["Embarked"].fillna(mostFreqPort)
#print(d[["Embarked"]].info())

fMed = d["Fare"].median()
d.loc[d["Fare"]==0,"Fare"] = fMed

#d["FareBand"] = pd.qcut(d["Fare"],5)
#quntiles = 4,8,11.5,22,40,515
#print(d[["FareBand","Fare"]].groupby("FareBand").mean())
d.loc[d["Fare"]<=8,"Fare"] = 0
d.loc[(d["Fare"]>8) & (d["Fare"]<=11.5),"Fare"] = 1
d.loc[(d["Fare"]>11.5) & (d["Fare"]<=22),"Fare"] = 2
d.loc[(d["Fare"]>22) & (d["Fare"]<=40),"Fare"] = 3
d.loc[(d["Fare"]>40) & (d["Fare"]<=515),"Fare"] = 4
#print(d["Fare"].head(10))

trainP = 0.8
tA = int(trainP*d.shape[0])

X = d.drop(["Survived","Title"], axis=1)
XTrain = X.iloc[:tA]
XTest = X.iloc[tA:]

Y = d["Survived"]
YTrain = Y.iloc[:tA]
YTest = Y.iloc[tA:]


logReg = linear_model.LogisticRegression()
logReg.fit(XTrain,YTrain)
print(logReg.score(XTest,YTest))

lrc = pd.DataFrame()
lrc["Names"] = X.columns
lrc["Coefficients"] = pd.Series(logReg.coef_[0])
print(lrc.sort_values(by="Coefficients",axis=0,ascending=False))

