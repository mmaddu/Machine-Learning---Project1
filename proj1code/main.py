#Project Members

print("\nUbitName1: nshokeen")
print("personNumber1: 50247681")


print("\nUbitName2: mmaddu")
print("personNumber2: 50246769")


print("\nUbitName3: csudhars")
print("personNumber3: 50245956")


#Importing Required Libraries

import scipy as scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from scipy import stats, integrate
import seaborn as sns

#Creating a data frame from the given data

df = pd.read_excel('/Users/mahalakshmimaddu/Desktop/IML/DataSet3/universitydata.xlsx', sheetname = 'university_data')

#Creating a data frame for the first four columns of the data

df1 = pd.DataFrame(df,columns=['CS Score (USNews)','Research Overhead %','Admin Base Pay$','Tuition(out-state)$'])



#Calculating Mean, Variance and Standard Deviation for the first for columns

mu1 = df['CS Score (USNews)'].mean()
mu2 = df['Research Overhead %'].mean()
mu3 = df['Admin Base Pay$'].mean()
mu4 = df['Tuition(out-state)$'].mean()



var1 = df['CS Score (USNews)'].var()
var2 = df['Research Overhead %'].var()
var3 = df['Admin Base Pay$'].var()
var4 = df['Tuition(out-state)$'].var()



sigma1 = df['CS Score (USNews)'].std()
sigma2 = df['Research Overhead %'].std()
sigma3 = df['Admin Base Pay$'].std()
sigma4 = df['Tuition(out-state)$'].std()



#Finding Correlation and Covariance Matrix

covarianceMat = cov = df1.cov()
correlationMat = corr = df1.corr()




#Printing the values

print ("\nmu1 = "+str(round(mu1,3)))
print ("mu2 = "+str(round(mu2,3)))
print ("mu3 = "+str(round(mu3,3)))
print ("mu4 = "+str(round(mu4,3)))

print ("\nvar1 = "+str(round(var1,3)))
print ("var2 = "+str(round(var2,3)))
print ("var3 = "+str(round(var3,3)))
print ("var4 = "+str(round(var4,3)))

print ("\nsigma1 = "+str(round(sigma1,3)))
print ("sigma2 = "+str(round(sigma2,3)))
print ("sigma3 = "+str(round(sigma3,3)))
print ("sigma4 = "+str(round(sigma4,3)))

print("\n")

print("covarianceMat = \n%s" %covarianceMat)

print("\n")

print("correlationMat = \n%s" %covarianceMat)

print("\n")




#Finding the loglikelihood (Univariate)

import math

mul1 = 1
for row_idx1 in range(0,np1.size - 1):
    mul1 = mul1 * stats.norm.pdf(np1[row_idx1],mu1,sigma1)

mul2 = 1
for row_idx2 in range(0,np2.size - 1):
    mul2 = mul2 * stats.norm.pdf(np2[row_idx2],mu2,sigma2)

mul3 = 1
for row_idx3 in range(0,np3.size - 1):
    mul3 = mul3 * stats.norm.pdf(np3[row_idx3],mu3,sigma3)
    
mul4 = 1
for row_idx4 in range(0,np4.size - 1):
    mul4 = mul4 * stats.norm.pdf(np4[row_idx4],mu4,sigma4)
  
  
totmul = mul1 * mul2 * mul3 * mul4

logLikelihood1 = math.log(mul1) + math.log(mul2) + math.log(mul3) + math.log(mul4)

print("\nlogLikelihood(Univariate) = %s" %logLikelihood1)


#Finding the loglikelihood (Multivariate)


from scipy.stats import multivariate_normal as mvnorm

multiVariate = 0

mu_Vector = [round(mu1,3),round(mu2,3),round(mu3,3),round(mu4,3)]

for itr in range(0,49):
   multiVariate = multiVariate + math.log(scipy.stats.multivariate_normal.pdf(df1.iloc[itr,:],mu_Vector,cov,allow_singular=True))

print("\nlogLikelihood (Multivariate) = %s" % multiVariate)


#converting dataframe into matrix
np1 = df1.as_matrix(columns=df1.columns[0:1])
np2 = df1.as_matrix(columns=df1.columns[1:2])
np3 = df1.as_matrix(columns=df1.columns[2:3])
np4 = df1.as_matrix(columns=df1.columns[3:4])


print("\n\n\nQuestion2 :Scatter Plots\n")


#Pairwise Scatter Plots 

print("\n")
print("Scatter Plot for CS Score (USNews) VS Research Overhead %")
sns.regplot(x="CS Score (USNews)", y="Research Overhead %", data=df1)
plt.show()
print("\n")
print("Scatter Plot for CS Score (USNews) VS Admin Base Pay$")
sns.regplot(x="CS Score (USNews)", y="Admin Base Pay$", data=df)
plt.show()
print("\n")
print("Scatter Plot for CS Score (USNews) % VS Tuition(out-state)$")
sns.regplot(x="CS Score (USNews)", y="Tuition(out-state)$", data=df1)
plt.show()
print("\n")
print("Scatter Plot for Research Overhead % VS Admin Base Pay$")
sns.regplot(x="Research Overhead %", y="Admin Base Pay$", data=df1)
plt.show() 
print("\n")
print("Scatter Plot for Research Overhead % VS Tuition(out-state)$")
sns.regplot(x="Research Overhead %", y="Tuition(out-state)$", data=df1)
plt.show()
print("\n")
print("Scatter Plot for Admin Base Pay$ VS Tuition(out-state)$")
sns.regplot(x="Admin Base Pay$", y="Tuition(out-state)$", data=df1)
plt.show()
print("\n")



#Creating Scatter matrix for Correlation



fig = plt.figure()
x1 = fig.add_subplot(111)
x2 = x1.matshow(corr,vmin = -1,vmax = 1)
fig.colorbar(x2)


x3 = ['CS Score','Research Overhead %','Admin Base Pay$','Tuition(out-state)$']

x1.set_xticklabels(['']+x3)
x1.set_yticklabels(['']+x3)



print("\nScatter Matrix for Correlation Matrix:\n")
plt.show()
print("\n")






