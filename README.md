Census Income Prediction
===

This python script that predicts whether a person's income excedes $50K/yr given census data. The attributes include age, workclass, education, race, sex, capital gain, etc. 

This demo uses the scikit-learn, pandas, numpy, and matplotlib libraries. The algorithms used in the model were K Nearest Neighbors, Decision tree, XGBoost, and CatBoost.  

Analysis
---

First, we have to clean the data and and change categorical variables to numeric. Start off by checking for any missing values:

```python
print(df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')
```  
Luckily, there is no missing data.  

After that we have to change all categorical values and set as numeric. We can do this by assigning each value a number. We start off with the income variable. If the the value is `<=50K`, we assign it a 0 and a 1 if it is `>50K`. These new values will be assigned into the new `income_level` variable:  

```python
df['income_level'] = np.where(df.income == '<=50K', 0, 1)
```  
After that, we change the `sex` variable and assign its values to the new `gender` variable:  

```python
df['gender'] = df['sex'].map({'Male':0, 'Female':1}).astype(int)
```

Now do the `race` variable, assigning a number to each category and inserting it into the new `ethnicity` variable:  

```python
ethnicity_key = {'White':0, 'Black':1, 'Asian-Pac-Islander':2,
'Amer-Indian-Eskimo':3, 'Other':4}

df['ethnicity'] = df['race'].map(ethnicity_key).astype(int)
```  

Same thing with the `native.country` variable. Assign new numeric values to the new `native_country` variable.  

```python
origin_key = {'?':0,'United-States':1, 'Mexico':2, 'Philippines':3,
'Germany':4, 'Canada':5, 'Puerto-Rico':6, 'El-Salvador':7, 
'India':8, 'Cuba':9, 'England':10,'Jamaica':11, 'South':12, 
'China':13, 'Italy':14, 'Dominican-Republic':15, 'Vietnam':16,
'Guatemala':17, 'Japan':18, 'Poland':19, 'Columbia':20, 'Taiwan':21,
'Haiti':22, 'Iran':23, 'Portugal':24, 'Nicaragua':25, 'Peru':26, 
'France':27, 'Greece':28, 'Ecuador':29, 'Ireland':30,'Hong':31,
'Trinadad&Tobago':32, 'Cambodia':33, 'Laos':34, 'Thailand':35, 
'Yugoslavia':36, 'Outlying-US(Guam-USVI-etc)':37, 'Hungary':38,
'Honduras':39, 'Scotland':40, 'Holand-Netherlands':41}

df['native_country'] = df['native.country'].map(origin_key).astype(int)

```
 Now do the `workclass` variable and assign new values to the new `work` variable:  

```python
 work_key = {'Private':0, 'Self-emp-not-inc':1, 'Local-gov':2, '?':3, 
'State-gov':4, 'Self-emp-inc':5, 'Federal-gov':6, 
'Without-pay':7,'Never-worked':8}

df['work'] = df['workclass'].map(work_key).astype(int)
```  

Now do the `marital.status` variable and assign new values to the new `marital_status`variable:  

```python
marital_status_key = {'Married-civ-spouse':0, 'Never-married':1, 'Divorced':2,
'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 
'Married-AF-spouse':6}

df['marital_status'] = df['marital.status'].map(marital_status_key).astype(int)
```  

Do the same with the `occupation` variable:  

```python
occupation_key = {'Prof-specialty':0, 'Craft-repair':1, 'Exec-managerial':2, 
'Adm-clerical':3, 'Sales':4, 'Other-service':5,
'Machine-op-inspct':6, '?':7, 'Transport-moving':8, 
'Handlers-cleaners':9, 'Farming-fishing':10, 'Tech-support':11,
'Protective-serv':12, 'Priv-house-serv':13, 'Armed-Forces':14}

df['occupation'] = df['occupation'].map(occupation_key).astype(int)
```  
Lastly, do the same wtih the `relationship` variable:  

```python
relationship_key = {'Husband':0, 'Not-in-family':1, 'Own-child':2, 'Unmarried':3,
'Wife':4, 'Other-relative':5}

df['relationship'] = df['relationship'].map(relationship_key).astype(int)
```  
Drop most of the original variables that we don't need anymore:


```python
df = df.drop(['income'], axis = 1)
df = df.drop(['sex'], axis = 1)
df = df.drop(['race'], axis = 1)
df = df.drop(['native.country'], axis = 1)
df = df.drop(['workclass'], axis = 1)
df = df.drop(['marital.status'], axis = 1)
df = df.drop(['education'], axis = 1)
```  

Now in order to use the `hours.per.week` variable, it will be easier to group the values into sections. These sections will be `< 40`,`== 40`, and `> 40`:

```python
df['hours.per.week'] = df['hours.per.week'].astype(int)
df.loc[df['hours.per.week'] < 40, 'hours.per.week'] = 0
df.loc[df['hours.per.week'] == 40, 'hours.per.week'] = 1
df.loc[df['hours.per.week'] > 40, 'hours.per.week'] = 2
```  

Finally, we are done cleaning the data. We can move on to visualizing it and making graphs. We will start by printing a frequency table of people's education levels:

```python
print(df['education.num'].value_counts())
```  

Get an idea of the population's age by graphing an histogram of the `age` variable:

![ageHist.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/ageHist.png)




Sources and Helpful Links
---  
https://cseweb.ucsd.edu/~jmcauley/cse190/reports/sp15/048.pdf
https://www.kaggle.com/uciml/adult-census-income
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/  
https://www.kaggle.com/kanav0183/catboost-and-other-class-algos-with-88-accuracy