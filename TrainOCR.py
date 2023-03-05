import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle
from skimage import io
from skimage.filters import threshold_otsu
from skimage.transform import resize,rescale
import os

CHARACTERS = ["0","1","2","3","4","5","6","7","8","9","B","C","P"]

def readTrainingData(imageDataDirectory):

    imageData = []
    targetData = []
    for each_letter in CHARACTERS:

        for imageIndex in range(950):
            imageIndex = imageIndex + 1
            imagePath = os.path.join(imageDataDirectory,each_letter,str(imageIndex)+".jpg")
            print(imagePath)
            readImage2Gray = io.imread(imagePath,as_gray=True)
            resizedImage = resize(readImage2Gray,[30,30])
            thresholdImage = resizedImage < threshold_otsu(resizedImage)
            # converting image to 1D array
            flatBinaryImage = np.array(thresholdImage).reshape(-1)
            imageData.append(flatBinaryImage)
            targetData.append(each_letter)


    return np.array(imageData), np.array(targetData)

def crossVerification(model,train_data,train_target,no_of_verifications_fold):

    testResult = cross_val_score(model,train_data,train_target,cv=no_of_verifications_fold)
    print("cross validation : FOLD ",str(no_of_verifications_fold))
    print(testResult*100)



dirCurrent = './images'
imgData,tarData = readTrainingData(dirCurrent)
svc_model = SVC(kernel='linear',probability=True)
crossVerification(svc_model,imgData,tarData,3)

#Train the model
svc_model.fit(imgData,tarData)

#Save model into file , ( serializing the objects with pickle )
print("saving model")
modelPath = "./dataModelNEWFINAL.sav"
pickle.dump(svc_model,open(modelPath,"wb"))
print("model saved")

print(imgData)
print(tarData)

