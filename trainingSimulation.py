print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from utils import *


#step 1
path = 'myData'
data = importDataInfo(path)

#step 2 Data visualization and balancing
balanceData(data,display=False)

#step 3
imagesPath, steerings = loadData(path,data)

#step 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#step 5


#step 6


#step 8
model = createModel()
model.summary()


#step 9
history=model.fit(batchGen(xTrain,yTrain,150,1), steps_per_epoch=300,epochs=50,
                  validation_data=batchGen(xVal,yVal,150,0),validation_steps=200)

#step 10
model.save('model.h5')
print("Model is saved")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 0.3])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
