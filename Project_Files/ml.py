import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

numAtts = 10
data = np.loadtxt("data.txt")
np.random.shuffle(data)
numTraining = int(len(data)*0.75)

trainingData = data[:numTraining]
testData = data[numTraining:]

c = RandomForestClassifier(n_estimators=100,max_features=0.6,bootstrap=True)

x = trainingData[:,:numAtts]
y = trainingData[:,numAtts]

c.fit(x,y)

print("Training Data Score: %f"%c.score(x,y))

x = testData[:,:numAtts]
y = testData[:,numAtts]

print("Test Data Score %f"%c.score(x,y))



