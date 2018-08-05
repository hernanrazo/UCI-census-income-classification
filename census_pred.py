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

print(df.describe())
print(' ')

#print frequency table of population's education level
print(df['education.num'].value_counts())

#make a boxplot showing people's income based on age
edIncomeBoxPlot = plt.figure()
plt.title('Income Based on Age')
df.boxplot(column = 'age', by = 'income')
plt.savefig(graph_folder_path + 'edIncomeBoxPlot')





