Census Income Prediction
===

This python script that predicts whether a person's income excedes $50K/yr given census data. The attributes include age, workclass, education, race, sex, capital gain, etc. This exercise is part of the [Adult Census Income](https://www.kaggle.com/uciml/adult-census-income) challenge on Kaggle.

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
Get an idea of the population's age by graphing a histogram of the `age` variable:  

![ageHist.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/ageHist.png)

Next, make various graphs displaying the correlation between income and other variables. Start with gender:  

![incomeGender.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/incomeGenderBar.png)  

We can see that men are more likely to be both above $50K and below $50K. We can also see that women are more likely to have an income of less than $50K.

Next, let's look at education:  

![incomeEdBar.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/incomeEdBar.png)   

This tells us that those that had less than a 12th grade education have a very slim chance of making more than $50K. It also tells us that an education of at least a high school diploma drastically increases your chances of making more than $50K. This also shows us that once we get to higher education levels past a bachelors, there are more people with incomes higher than 50K than those below.

Now let's take a look at occupation. Plot a graph showing income based on the listed occupations. 

![incomeOccGraph.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/incomeOccGraph.png)  

This graph tells us that most people with high incomes had jobs in the `Prof-speciality`, `Craft-repair`, `Exec-managerial`, `Adm-clerical`, and `Sales` categories. `Prof-speciality` and `Exec-managerial` especially had a high amount of people with incomes above 50K. We can also see that those in the `priv-house-service`are gauranteed to not earn above 50K. 

`Relationship` is the next variable we'll look at. Make a bar graph correlating this with income.  

![incomeMarriageBar.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/incomeMarriageBar.png)  

We can see that those in `Husband` have most of the higher incomes compared to the other categories but still have almost the same amount in the lower income level. Those in the `Unmarried`, `Wife`, and `Other-relative` categories have mosty everyone in the less than $50K.  

![incomeCountryBar.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/incomeCountryBar.png)  

Lastly, let's look at the hours per week variable. make a bar graph comparing hours per week and income:

![incomeHoursBar.png](https://github.com/hrazo7/UCI-census-income-classification/blob/master/graphs/incomeHoursBar.png)  

Since we split the data into three categories earlier, there are only three x-values. We can see that those that made more than $50K worked 40 or more hours per week. a very small amount of those with high incomes worked less than 40 hours per week.  

Now we can start working on some models. We will try a few and stick with the one that got the best score.  

Start off with K-nearest Neighbors algorithm. We will use a range of 1 to 25 for the value of n. Store each score in an empty list and return the best score after training the model.  

```python
k_values = np.arange(1, 25)
scores = []

for k in k_values:
	model = KNeighborsClassifier(n_neighbors = k)
	model.fit(train_x, train_y)
	KNN_prediction = model.predict(test_x)
	scores.append(metrics.accuracy_score(test_y, KNN_prediction))

print('KNN Results:')
print(scores.index(max(scores)), max(scores))
print(' ')
```  

The best accuracy score from this algorithm is 0.7917 when n = 21.  

Let's try the decision tree algorithm:  

```python
model = DecisionTreeClassifier(class_weight = None, min_samples_leaf = 100, 
	random_state = 10)
model.fit(train_x, train_y)

DTC_prediction = model.predict(test_x)

print('Decision tree Results:')
print(metrics.accuracy_score(test_y, DTC_prediction))
print(' ')
```  

This algorithm returns an accuracy score of 0.8568. That's better than the KNN model, but let's try the XGBoost algorithm:  


```python
XGBClassifier = XGBClassifier()
XGBClassifier.fit(train_x, train_y)

XGBC_prediction = XGBClassifier.predict(test_x)

print('XGBoost results:')
print(metrics.accuracy_score(test_y, XGBC_prediction))
```  
We get an accuracy score of 0.8691.  

Let's try the catboost algorithm:  

```python
CBClassifier = CatBoostClassifier(learning_rate = 0.04)
CBClassifier.fit(train_x, train_y)
CBC_prediction = CBClassifier.predict(test_x)

print('CatBoost results:')
print(metrics.accuracy_score(test_y, CBC_prediction))
```  

This algorithm returns an accuracy score of 0.873.

Acknowledgements
---

I reviewed various Kaggle kernels and other sources. One resource I used was [this paper by Chet Lemon, Chris Zelazo, and Kesav Mulakaluri](https://cseweb.ucsd.edu/~jmcauley/cse190/reports/sp15/048.pdf). I also reviewed [this Kaggle Kernel by Kanavanand](https://www.kaggle.com/kanav0183/catboost-and-other-class-algos-with-88-accuracy).

Sources and Helpful Links
---  
https://cseweb.ucsd.edu/~jmcauley/cse190/reports/sp15/048.pdf
https://www.kaggle.com/uciml/adult-census-income
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/  
https://www.kaggle.com/kanav0183/catboost-and-other-class-algos-with-88-accuracy