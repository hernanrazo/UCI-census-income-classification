import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

#get dataset
df = pd.read_csv('/Users/hernanrazo/pythonProjects/census_income_prediction/adult.csv')

#make a string that holds most of the folder path
graph_folder_path = '/Users/hernanrazo/pythonProjects/census_income_prediction/graphs/'

#start off data analysis by searching for missing values
print(df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')

#there are a lot of categorical variables in the
#dataset. Start off by turning them into numerical values so 
#plotting the data can be a little easier

#switch income values so they are numerical
df['income_level'] = np.where(df.income == '<=50K', 0, 1)

#switch sex variables so they are numerical
df['gender'] = df['sex'].map({'Male':0, 'Female':1}).astype(int)

#convert the race variable into numeric form by
#assigning each with a numeric value
ethnicity_key = {'White':0, 'Black':1, 'Asian-Pac-Islander':2,
			     'Amer-Indian-Eskimo':3, 'Other':4}

df['ethnicity'] = df['race'].map(ethnicity_key)

#do the same with the native.country variable
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

df['native_country'] = df['native.country'].map(origin_key)

#change the workclass variable into numerical by
#assigning each a unique number
work_key = {'Private':0, 'Self-emp-not-inc':1, 'Local-gov':2, '?':3, 
			'State-gov':4, 'Self-emp-inc':5, 'Federal-gov':6, 
			'Without-pay':7,'Never-worked':8}

df['work'] = df['workclass'].map(work_key)

#change the marital.status variable into numeric
marital_status_key = {'Married-civ-spouse':0, 'Never-married':1, 'Divorced':2,
					  'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 
					  'Married-AF-spouse':6}

df['marital_status'] = df['marital.status'].map(marital_status_key)

#now do the occupation variable
occupation_key = {'Prof-specialty':0, 'Craft-repair':1, 'Exec-managerial':2, 
				  'Adm-clerical':3, 'Sales':4, 'Other-service':5,
				  'Machine-op-inspct':6, '?':7, 'Transport-moving':8, 
				  'Handlers-cleaners':9, 'Farming-fishing':10, 'Tech-support':11,
				  'Protective-serv':12, 'Priv-house-serv':13, 'Armed-Forces':14}

df['occupation'] = df['occupation'].map(occupation_key)

#do the same wtih the realtionship variable
relationship_key = {'Husband':0, 'Not-in-family':1, 'Own-child':2, 'Unmarried':3,
'Wife':4, 'Other-relative':5}

df['relationship'] = df['relationship'].map(relationship_key)

#do the same with the race variable
#drop most original variables
df = df.drop(['income'], axis = 1)
df = df.drop(['sex'], axis = 1)
df = df.drop(['race'], axis = 1)
df = df.drop(['native.country'], axis = 1)
df = df.drop(['workclass'], axis = 1)
df = df.drop(['marital.status'], axis = 1)

#place hours per week into three categories: <40, 40, amd >40
df['hours.per.week'] = df['hours.per.week'].astype(int)
df.loc[df['hours.per.week'] < 40, 'hours.per.week'] = 0
df.loc[df['hours.per.week'] == 40, 'hours.per.week'] = 1
df.loc[df['hours.per.week'] > 40, 'hours.per.week'] = 2

print(df['hours.per.week'].value_counts())

#start visualizing the freshly cleaned data

#print frequency table of population's education level
print(df['education.num'].value_counts())

#make a histogram showing people's age 
ageHist = plt.figure()
plt.title('Age')
df['age'].hist(bins = 20)
plt.savefig(graph_folder_path + 'ageHist.png')

#make a bar graph showing income based on gender
incomeGenderBar = plt.figure()
incomeGenderBar = pd.crosstab(df['gender'], df['income_level'])
incomeGenderBar.plot(kind = 'bar', color = ['red','green'], 
	grid = False, title = 'Income Based on Gender')
plt.savefig(graph_folder_path + 'incomeGenderBar.png')

#make a bar graph showing income based on education
incomeEdBar = plt.figure()
incomeEdBar = pd.crosstab(df['education.num'], df['income_level'])
incomeEdBar.plot(kind = 'bar', color = ['red','green'], 
	grid = False, title = 'Income Based on Education')
plt.savefig(graph_folder_path + 'incomeEdBar.png')

#make a bar graph based on income and occupation
incomeGenderOccGraph = plt.figure()
incomeGenderOccGraph = pd.crosstab(df['occupation'], df['income_level'])
incomeGenderOccGraph.plot(kind = 'bar', stacked = True, color = ['red', 'blue'],
	grid = False, title = 'Income based on Occupation')
plt.savefig(graph_folder_path + 'incomeOccGraph.png')

#make a bar graph based on income and marital status
incomeMarriageBar = plt.figure()
incomeMarriageBar = pd.crosstab(df['marital_status'], df['income_level'])
incomeMarriageBar.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Income Based on Marital Status')
plt.savefig(graph_folder_path + 'incomeMarriageBar.png')

#make a bar graph based on income and native country
incomeCountryBar = plt.figure()
incomeCountryBar = pd.crosstab(df['native_country'], df['income_level'])
incomeCountryBar.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Income Based on Native Country')
plt.savefig(graph_folder_path + 'incomeCountryBar.png')

#make a graph based on income and hours worked per week
incomeHoursBar = plt.figure()
incomeHoursBar = pd.crosstab(df['hours.per.week'], df['income_level'])
incomeHoursBar.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Income Based on Hours per Week')
plt.savefig(graph_folder_path + 'incomeHoursBar.png')







