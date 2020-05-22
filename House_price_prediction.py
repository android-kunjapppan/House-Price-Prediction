import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#loading the dataset
dataset_train=pd.read_csv("train.csv")

#Description of the Dependant Variable
dataset_train['SalePrice'].describe()

#histogram of the dependant variable
sns.distplot(dataset_train['SalePrice'])

#skewness and Kurtosis
print("Skewness= %f" % dataset_train['SalePrice'].skew())
print("kurtosis= %f" % dataset_train['SalePrice'].kurt())



## Relationshipo between Numerical Variables
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([dataset_train['SalePrice'], dataset_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([dataset_train['SalePrice'], dataset_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



## Relationship between categorical Features
#Box plot overallauall/saleprice
var = 'OverallQual'
data = pd.concat([dataset_train['SalePrice'], dataset_train[var]], axis=1)
f, ax= plt.subplots(figsize=(8,6))
fig= sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)

#Box plot yearbuilt/saleprice
var = 'YearBuilt'
data = pd.concat([dataset_train['SalePrice'], dataset_train[var]], axis=1)
f, ax= plt.subplots(figsize=(8,6))
fig= sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000);
plt.xticks(rotation=90);



#Correlation Matrix(Heat Map style)
corrmat=dataset_train.corr()
f, ax=plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)


#Salesprice correlation matrix(Zoomed heat map style)
k=10 #Number if variables for the heat map
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm=np.corrcoef(dataset_train[cols].values.T)
sns.set(font_scale=1.25)
hm=sns.heatmap(cm, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values,xticklabels=cols.values)
plt.show()




##ScatterPlots
sns.set()
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(dataset_train[cols], size=2.5)
plt.show()



##Missing Data
total=dataset_train.isnull().sum().sort_values(ascending=False)
percent=(dataset_train.isnull().sum()/dataset_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20) 

##Handling the missing data
dataset_train=dataset_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
dataset_train=dataset_train.drop(dataset_train.loc[dataset_train['Electrical'].isnull()].index)
dataset_train.isnull().sum().max()  # checking that there is no data missing




##Univariate Analysis
#Standardize the data - data standardization means converting data values to have mean of 0 and a standard deviation of 1.
saleprice_scaled= StandardScaler().fit_transform(dataset_train['SalePrice'][:,np.newaxis])
low_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print(low_range)
print(high_range) #high range lies between 0-7 except the last two..    we can define them as outliers and delete them.


#deleting points
dataset_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
dataset_train = dataset_train.drop(dataset_train[dataset_train['Id'] == 1299].index)
dataset_train = dataset_train.drop(dataset_train[dataset_train['Id'] == 524].index)



#histogram and normal distribution plot for saleprice
sns.distplot(dataset_train['SalePrice'], fit = norm)
fig=plt.figure()
res= stats.probplot(dataset_train['SalePrice'],plot=plt) #saleprice is not normal.. it's showing positive skewedness(Peakedness). BUt logarithmic transformations can solve this problem.

#Logarithmic Transoformation saleprice
dataset_train['SalePrice']=np.log(dataset_train['SalePrice'])

#histogram and normal distribution plot after logarithmic transformation
sns.distplot(dataset_train['SalePrice'], fit = norm)
fig=plt.figure()
res= stats.probplot(dataset_train['SalePrice'],plot=plt)



#histogram and normal distribution plot for GrLivArea
sns.distplot(dataset_train['GrLivArea'], fit = norm)
fig=plt.figure()
res= stats.probplot(dataset_train['GrLivArea'],plot=plt) # it'll show skeawness

#Logarithmic Transoformation Grlivearea
dataset_train['GrLivArea']=np.log(dataset_train['GrLivArea'])


#histogram and normal distribution plot for GrLivArea after logarithmic transformation
sns.distplot(dataset_train['GrLivArea'], fit = norm)
fig=plt.figure()
res= stats.probplot(dataset_train['GrLivArea'],plot=plt)


#histogram and normal distribution plot for totalbsmtsf
sns.distplot(dataset_train['TotalBsmtSF'], fit = norm)
fig=plt.figure()
res= stats.probplot(dataset_train['TotalBsmtSF'],plot=plt) #Here the zerp values ddoesnt allow us to do logarithmic transformations



#removing all the unnecessarty variables
dataset_train=dataset_train.loc[:,cols]

#seperating dependant  and indepnedent variables
X=dataset_train.iloc[:,1:]
y=dataset_train.iloc[:,:1]

#test train split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#standardising the data
from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
X_train=Sc_X.fit_transform(X_train)
X_test=Sc_X.transform(X_test)

#Creating the Linear Regression model object
linear= linear_model.LinearRegression()
#Train the model. using the trainig dataset\
linear.fit(X_train,y_train)
#Make the predictionss using the testing set
y_pred=linear.predict(X_test)



#evalutaion Metrucs
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)


#plotting the ypred and ytest
plt.scatter(y_pred, y_test,  color='gray')
plt.show()








