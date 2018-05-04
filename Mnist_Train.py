import numpy as np
import pandas as pd 
import time,pickle
import cv2 as cv
from PIL import Image

""" ML Models """
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
#from xgboost.sklearn import XGBClassifier
""" Scalers """
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,MaxAbsScaler,Binarizer,RobustScaler,LabelEncoder

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


def showData(itemList):
    """ Display image from vector (1,784) """
    allInRow = ''.join(['.' if i==0 else "#" for i in itemList])
    allInRowWithSplitter = ''.join(["/" if i%28==0 else allInRow[i] for i in range(len(allInRow))])
    rows = allInRowWithSplitter.split('/')
    for row in rows:
        print(row)


def getImageVectorOpencv(path):
    """ get Image Vector using opencv along with Numpy """
    image = cv.imread(path)
    imageGray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    resizedImage = np.resize(imageGray,(28,28))
    imageVectorNp = np.reshape(resizedImage,(1,784))    
    vec = [0 if px==255 else px for px in imageVectorNp.tolist()[0]]
    return vec

def getImageVectorPil(path):
    """ get Image Vector using PIL"""
    original_image = Image.open(path)    
    size = (28,28)
    imageResized = original_image.resize(size)    
    grayImage = imageResized.convert("L")
    imageData = list(grayImage.getdata())
#    resized_image.getchannel(0).show()
    vec = [0 if px==255 else 1 if px==0 else px for px in imageData]
    showData(vec)
    return vec
                
def getDataLinux():
    """ Get Data """
    from mnist import MNIST
    mndata = MNIST('../Mnist_data/')
    mndata.gz = True
    trainImages, trainLabels = mndata.load_training()
    testImages, testLabels = mndata.load_testing()
    return trainImages, trainLabels

def getDataWindows():
    """ Get Data """
    import mnist     
    trainDataImages = mnist.train_images()
    trainDataLabels = mnist.train_labels()
    testDataImages = mnist.test_images()
    testDataLabels = mnist.test_labels()
    
    trainImages = [img for img in trainDataImages]
    trainVectors = [np.reshape(img,(1,784)) for img in trainImages]
    trainDataVectors = np.zeros((len(trainVectors),len(trainVectors[0][0])))
    for i in range(len(trainVectors)):
        trainDataVectors[i] = np.array(trainVectors[i][0])
    return trainDataVectors, trainDataLabels


def getXY(train,label):
    """ define trained Data and Labels """
    data = pd.DataFrame(np.array(train))
    x = data
    y = np.array(label)
    return x,y

def checkAccuracy(X,Y,model):
    """ Checking the accuracy of Random Forest"""
    prediction = np.zeros(Y.shape[0])
    skf = StratifiedKFold(n_splits=5)
    
    for train,test in skf.split(X,Y):
        x_train = X.iloc[train,:]
        x_test = X.iloc[test,:]
        y_train = Y[train]
        y_test = Y[test]
        
        model.fit(x_train,y_train)
#        print(type(x_test))
        result = model.predict(x_test)
#        print(result,type(result),"\n")
        prediction[test] = result
    print("\aAccuracy: %.8f " % accuracy_score(Y,prediction))
    #Random Forest with MinMaxScaler --> acc 0.96713333
    #Random Forest without scaling   --> acc 0.96715000

def predictImage(path,model):
    """ testing on an Image """
    imageVector = getImageVectorPil(path)
    
    df = pd.DataFrame(index=[0],columns=[i for i in range(784)])
    for i in range(len(imageVector)):
        df.iloc[:,i] = imageVector[i]
    
    prediction = model.predict(df)
    output = prediction[0]
    return output

def saveModel(model,fileName):
    ''' Saving the model '''
    pickle.dump(model, open(fileName, 'wb'))    

def loadModel(fileName):
    ''' loading the model '''
    return pickle.load(open(fileName, 'rb'))

def main():
    
    randomForest = RandomForestClassifier(n_estimators=170, random_state=42,n_jobs=-1)
    
    """ training Random Forest Model"""
    print("Getting Data ... ")
    train,label = getDataWindows()
    x,y = getXY(train,label)
    
#    print("Fitting the Model ...")
#    #checkAccuracy(x,y,randomForest) #optional
#    randomForest.fit(x,y)
    
#    print("Saving the Model ...")
    filename = 'Random_Forest_model_with_Mnist_data.sav'
#    saveModel(randomForest,filename)
    
    print("Loading the Model ...")
    ModelLoaded = loadModel(filename)
    
    print("Check an Image ...")
    imagePath = r"C:\Users\HatemZam\Desktop\TestNumbers\6.jpg"
    action = predictImage(imagePath,ModelLoaded)
    print("predicted value: ",action)


if __name__ == "__main__":
    main()