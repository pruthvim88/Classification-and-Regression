import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def classificationMeanList(X,y):
    classes  = np.unique(y)
    classesList = []
    meansList = []
    for i in range(len(classes)):
        classesList.append(np.zeros((1,len(X[0]))))
    for i in range(len(y)):
        for j in range(len(classes)):
            if(y[i]==classes[j]):
                classesList[j] = np.vstack((classesList[j],X[i]))
    for i in range(len(classes)):
        classesList[i] = np.delete(classesList[i],0,0)
        meansList.append(np.mean(classesList[i],axis=0))
    return classesList, meansList

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    means = classificationMeanList(X,y)[1]
    covmat = np.cov(X,rowvar=0)              
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    classes  = classificationMeanList(X,y)[0]
    means = classificationMeanList(X,y)[1]
    covmats = []
    for i in range(len(classes)):
        covmats.append(np.cov(classes[i],rowvar=0)) 
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    D = len(means[0])
    covmatDet = np.linalg.det(covmat)
    covmatinv = np.linalg.inv(covmat)
    coef = 1/((np.power(2*pi,D/2))*np.sqrt(covmatDet))
    pdftestList = []
    for i in range(len(means)):
        pdftestList.append(np.zeros(1))
    pdf = []
    for i in range(len(means)):
        for j in range(len(Xtest)):
            xminusmu = Xtest[j]-means[i]
            exponent = np.exp(-np.dot(np.dot(xminusmu,covmatinv),np.transpose(xminusmu))/2)
            pdftestList[i]= np.vstack((pdftestList[i],coef*exponent))
        pdf.append(pdftestList[i])
    for i in range(len(pdf)):
        pdf[i] = np.delete(pdf[i],0,0)
    total = 0
    ypred = np.argmax(pdf, axis=0)+1
    for i in range(len(pdf[0])):
        if(ypred[i] == ytest[i]):
            total = total+1
    acc = (total/len(ytest))*100
    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    D = len(means[0])
    pdftestList = []
    for i in range(len(means)):
        pdftestList.append(np.zeros(1))
    pdf = []
    for i in range(len(means)):
        covmatDet = np.linalg.det(covmats[i])
        covmatinv = np.linalg.inv(covmats[i])
        coef = 1/((np.power(2*pi,D/2))*np.sqrt(covmatDet))
        for j in range(len(Xtest)):
            xminusmu = Xtest[j]-means[i]
            exponent = np.exp(-np.dot(np.dot(xminusmu,covmatinv),np.transpose(xminusmu))/2)
            pdftestList[i]= np.vstack((pdftestList[i],coef*exponent))
        pdf.append(pdftestList[i])
    for i in range(len(pdf)):
        pdf[i] = np.delete(pdf[i],0,0)
    total = 0
    ypred = np.argmax(pdf, axis=0)+1
    for i in range(len(pdf[0])):
        if(ypred[i] == ytest[i]):
            total = total+1
    acc = (total/len(ytest))*100
    # IMPLEMENT THIS METHOD
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    invX = np.linalg.inv(np.dot(np.transpose(X),X))
    w = np.dot(invX,np.dot(np.transpose(X),y))
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
    invpart = np.linalg.inv((lambd*np.identity(X.shape[1]))+np.dot(np.transpose(X),X))
    w = np.dot(invpart,np.dot(np.transpose(X),y))
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    sub = (ytest-np.dot(Xtest,w))
    mse = np.dot(np.transpose(sub),sub)/len(ytest)
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    w = w.reshape(w.shape[0],1)
    yminusXW = y-np.dot(X,w)
    regular = (lambd*np.dot(np.transpose(w),w))/2
    error = (np.dot(np.transpose(yminusXW),yminusXW)/2)+regular
    
    grad = -np.dot(np.transpose(y),X)+np.dot(np.transpose(w),np.dot(np.transpose(X),X))
    regulargrad = lambd*np.transpose(w)
    error_grad = grad+regulargrad
    
    return error, np.array(error_grad).flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
    Xd=np.zeros([x.shape[0],])
    Xd=np.transpose(np.matrix(Xd))
    for j in range(p+1):
        col=np.power(x,j)
        col=np.transpose(np.matrix(col))
        Xd=np.hstack((Xd,col))
    Xd = np.delete(Xd,0,1)

    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_tr = testOLERegression(w,X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_tr_i = testOLERegression(w_i,X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('===============Linear Regression===================')
print('For Training data:')
print('MSE without intercept '+str(mle_tr))
print('MSE with intercept '+str(mle_tr_i))

print('For Testing data:')
print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

x = np.linspace(0, 65, num=65)
plt.plot(x,w_i)
plt.title('Variation of weight components in OLE regression')
plt.show()

# Problem 3
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnRidgeRegression(X,y,1)
mle_tr = testOLERegression(w,X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnRidgeRegression(X_i,y,1)
mle_tr_i = testOLERegression(w_i,X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('===============Ridge Regression===================')
print('For Training data:')
print('MSE without intercept '+str(mle_tr))
print('MSE with intercept '+str(mle_tr_i))

print('For Testing data:')
print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

x = np.linspace(0, 65, num=65)
plt.plot(x,w_i)
plt.title('Variation of weight components in Ridge regression')
plt.show()

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 30}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses4)] # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
