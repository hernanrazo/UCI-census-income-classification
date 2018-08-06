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

#switch income values so they are numerical. '<=50k' is switched to 0 
#and '>50k' is switched to 1

income_key = {'<=50k':0, '>50k':1}

#for i in df:
#	i['income'] = i['income'].map(income_key)
df['income'] = df['income'].map(income_key)


print(df.head())









incomeCountryPie = plt.figure()
plt.pie(df['income'], labels = df['native.country'])
plt.savefig(graph_folder_path + 'incomeCountryPie.png')




'''
#print frequency table of population's education level
print(df['education.num'].value_counts())

#make a histogram showing people's age 
ageHist = plt.figure()
plt.title('Age')
df['age'].hist(bins = 20)
plt.savefig(graph_folder_path + 'ageHist.png')

#make a bar graph showing income based on gender
incomeGenderBar = plt.figure()
incomeGenderBar = pd.crosstab(df['sex'], df['income'])
incomeGenderBar.plot(kind = 'bar', color = ['red','green'], 
	grid = False, title = 'Income Based on Gender')
plt.savefig(graph_folder_path + 'incomeGenderBar.png')

incomeCountryPie = plt.figure()
plt.pie(df['income'], labels = df['native.country'])
plt.savefig(graph_folder_path + 'incomeCountryPie.png')


incomeEdBar = plt.figure()
incomeEdBar = pd.crosstab(df['education'], df['income'])
incomeEdBar.plot(kind = 'bar', color = ['red','green'], 
	grid = False, title = 'Income Based on Education')
plt.savefig(graph_folder_path + 'incomeEdBar.png')
'''



