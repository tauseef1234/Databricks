# Databricks notebook source
# MAGIC %md # Caravan EDA - MA755
# MAGIC Date: 02/27/2018 <br \> 
# MAGIC Author: MIMOZA MARKO (marko_mimo@bentley.edu), SNEHA ARORA (arora_sneh@bentley.edu), TAUSEEF AHMAD (ahmad_taus@bentley.edu)

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to explain characteristics of customers. The analyses start with exploring the dataset and selecting the 10 most important variables in explaining the response variable CARAVAN. In order to perform the dimensionality reduction we apply the random forest classifier. After acquiring the important variables, we explore the single and multiple variable summaries to find patterns.

# COMMAND ----------

# MAGIC %md ## 1. Dataset Description

# COMMAND ----------

# MAGIC %md The dataset consists of 87 attributes and 9822 observations. It is further divided into a training set (5822 observations) and a test set (4000 observations). Out of the 86 attributes, 2 are categorical, 83 are numerical and 1 is the response variable (Caravan Insurance Purchased) which indicates whether the customer purchase a caravan insurance policy or not. This is an imbalanced dataset as the target variable Caravan Insurance Purchased has more 0’s i.e. the customers did not purchase the insurance policy as compared to 1’s i.e. the customers did purchase the insurance policy.  Futhermore, this dataset is set up as groups and each observation represent a large sample size.
# MAGIC There are 86 variables, containing sociodemographic data (variables 1-43) and product ownership data (variables 44-86). The sociodemographic data is derived from zip codes. All customers living in areas with the same zip code have the same sociodemographic attributes. 

# COMMAND ----------

# MAGIC %md 
# MAGIC Each row of the dataset represents a group of customers. The following set of variables describe this group of customers:
# MAGIC - `MAANTHUI`: Number of houses 1 - 10 (in the group of customers)
# MAGIC - `MGEMOMV`: Avg size household 1 - 6 (in the group of customers)
# MAGIC - `MOSHOOFD`: Customer main type; see L2
# MAGIC - `MOSTYPE`: Customer Subtype; see L0
# MAGIC - `MGEMLEEF`: Avg age; see L1
# MAGIC 
# MAGIC See the documentation for the meaning of L0, L1 and L2. 
# MAGIC 
# MAGIC The remaining variables that start with "M" provide demographic data and contain integers between `0` and `9` that represent ranges of percentages with `0` representing 0% and `9` representing 100%. 
# MAGIC The other integers represent ranges of about 13% each. 
# MAGIC See the documentation for details. 
# MAGIC 
# MAGIC The variables that start with "A" or "P" provide information about the customers in that postal code. 
# MAGIC They contain integers between `0` and `9` that represent ranges of counts for that variable.
# MAGIC See the documentation for details. 
# MAGIC 
# MAGIC The `CARAVAN` variable is the target variable.
# MAGIC It is also a _count_ variable (as above) recording the number of mobile home policies in that postal code.

# COMMAND ----------

# MAGIC %md ##2. Load Libraries

# COMMAND ----------

# MAGIC %md Load the required libraries and check version numbers. 

# COMMAND ----------

import numpy             as np
import pandas            as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt
import seaborn as sns
np.__version__, pd.__version__, mpl.__version__, sns.__version__

# COMMAND ----------

# MAGIC %md ##3. Read Dataset

# COMMAND ----------

# MAGIC %md First check that the files exists where we expect it.

# COMMAND ----------

# MAGIC %sh ls /dbfs/mnt/group-ma755/data

# COMMAND ----------

# MAGIC %md Check that the file has a header and looks reasonable. 

# COMMAND ----------

# MAGIC %sh head /dbfs/mnt/group-ma755/data/caravan-insurance-challenge.csv

# COMMAND ----------

caravan_df = pd.read_csv('/dbfs/mnt/group-ma755/data/caravan-insurance-challenge.csv')
caravan_df.head()

# COMMAND ----------

# MAGIC %md Check Dataset

# COMMAND ----------

caravan_df.info()

# COMMAND ----------

caravan_df.columns

# COMMAND ----------

# MAGIC %md There are 87 variables and 9,822 observations in this dataframe. 

# COMMAND ----------

caravan_df.shape

# COMMAND ----------

# MAGIC %md The `'Origin'` variable contains information if the observation is train or test. 5,822 of the observations are train and 4,000 are tests. 

# COMMAND ----------

sum(caravan_df['ORIGIN']== 'train'), sum(caravan_df['ORIGIN']== 'test')

# COMMAND ----------

# MAGIC %md ##4. Variable Selection

# COMMAND ----------

# MAGIC %md As a first step in this section we choose which variables to keep for analyzing. We first check the correlation of the numerical variables, then from the ones that are highly correlated we only keep one variable per each pair. We apply random forest classifier to check the variable importance.

# COMMAND ----------

numerical_columns = [name for name in list(caravan_df.columns) if name not in ['ORIGIN', 'CARAVAN', 'MOSTYPE', 'MOSHOOFD']]

# COMMAND ----------

# MAGIC %md Apply the `corr()` method to obtain the correlation table for the numerical columns of the `caravan_df`. The output is a pandas dataframe that we store in `correlation_df`. 

# COMMAND ----------

correlation_df = caravan_df[numerical_columns].corr(method="pearson") # apply the corr() method to numerical columns 
correlation_df # pandas dataframe

# COMMAND ----------

# MAGIC %md Filter the `correlation_df` to find those variables whose absolute correlation value is greater than .7. The output is a pandas dataframe. 

# COMMAND ----------

correlation_df[abs(correlation_df) > .7 ]

# COMMAND ----------

# MAGIC %md Applying the `stack()` method to the to the above correlation dataframe will return all the pairs associated with the correlation magnitude. The default of the method is to remove the NaN values. Therefore, we get only the pairs with values. The output of `stack()` here is a pandas series. 

# COMMAND ----------

# return pd series with the pairs of variables and their correlation magnitude that satifies the filter.
correlation_df[abs(correlation_df) > .7 ].stack()  

# COMMAND ----------

# MAGIC %md The code below returns a lists of the indices with correlation magnitude higher than 0.7. The `index()` method returns the indices, which in this case are tuples. Check the above series for reference. 

# COMMAND ----------

correlated_pairs=list(correlation_df[abs(correlation_df) > .7 ].stack().index)
correlated_pairs

# COMMAND ----------

# MAGIC %md Filter the `correlated_pairs` to exclude the pairs with the same variable, since it is the correlation of that variable and itself. 

# COMMAND ----------

correlated_unique_pairs = [tuple for tuple in correlated_pairs if (tuple[0] != tuple[1])]
correlated_unique_pairs

# COMMAND ----------

# MAGIC %md The variables below has been eliminated because they were highly correlated with other variables. These variables do not provide incremental information about the response variable.

# COMMAND ----------

eliminated_variables =['MHKOOP', 'MZFONDS', 'PWALAND', 'PWAPART', 'PGEZONG', 'PBROM', 'PBYSTAND', 'PAANHANG', 'PWAOREG', 'PZEILPL', 'PFIETS',   'PMOTSCO',  'PTRACTOR',  'PWERKT',  'PPERSAUT', 'PWABEDR', 'PVRAAUT', 'PPERSONG', 'PBESAUT', 'PPLEZIER', 'MRELOV', 'PBRAND',  'PINBOED', 'PLEVEN', 'MFWEKIND', 'MGODGE', 'MFALLEEN', 'MOPLMIDD', 'MAUT1', 'MOPLHOOG']
len(eliminated_variables)

# COMMAND ----------

# MAGIC %md Below is displayed the list of variables that has been chosen from each pair of highly correlated variables.

# COMMAND ----------

selected_variables = ['MHHUUR', 'MZPART', 'AWALAND', 'AWAPART', 'AGEZONG', 'ABROM', 'ABYSTAND', 'AAANHANG', 'AWAOREG', 'AZEILPL', 'AFIETS', 'AMOTSCO',  
'ATRACTOR', 'AWERKT',  'APERSAUT',  'AWABEDR',  'AVRAAUT',  'APERSONG', 'ABESAUT', 'APLEZIER', 'MRELGE', 'ABRAND', 'AINBOED', 'ALEVEN', 'MGEMOMV', 'MGODPR', 'MFALLEEN', 'MOPLLAAG', 'MAUT0', 'MSKA']
len(selected_variables)

# COMMAND ----------

# MAGIC %md The final list of variables for random forest classifier includes the selected numerical variables above, the numerical variables which were not highly correlated with any other variable and the categorical variables. 

# COMMAND ----------

rand_forest_var = [ var for var in caravan_df.columns if var not in eliminated_variables and var not in ['CARAVAN', 'ORIGIN']]

# COMMAND ----------

len(rand_forest_var)

# COMMAND ----------

# MAGIC %md Divide the data into train and test using only the random forest variables.

# COMMAND ----------

x_train= caravan_df.loc[caravan_df['ORIGIN']=='train', rand_forest_var]
x_test = caravan_df.loc[caravan_df['ORIGIN']=='test', rand_forest_var]

# COMMAND ----------

# MAGIC %md Define the `y_train` to be used in random forest classifier.

# COMMAND ----------

y_train=caravan_df.loc[caravan_df['ORIGIN']=='train','CARAVAN']

# COMMAND ----------

# MAGIC %md We perform the random forest classifier on the selected variables 55 variables.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0) # max_depth the depth of the tree, random_state is the random number generator
clf.fit(x_train, y_train)
RandomForestClassifier(bootstrap=True, criterion='gini') # bootsrap sampling is done, gini because it is classification 

# COMMAND ----------

# MAGIC %md Display the feature importance.

# COMMAND ----------

clf.feature_importances_

# COMMAND ----------

# MAGIC %md Zip the columns with their corresponding feature importance and store it as a list of tuples in `variable_importance`. 

# COMMAND ----------

variable_importances = list(zip(x_train.columns,clf.feature_importances_))
variable_importances

# COMMAND ----------

# MAGIC %md Sort the above list according to the second item of the tuples which is the importance magnitude. 

# COMMAND ----------

sorted_var_importances = sorted(variable_importances, key=lambda x: x[1], reverse = True)
sorted_var_importances

# COMMAND ----------

# MAGIC %md Top 10 most important variables according to the random forest result. The last one is the response CARAVAN. 

# COMMAND ----------

most_imp_var= ['APERSAUT', 'MINKGEM', 'MKOOPKLA','MINKM30','AWAPART', 'MHHUUR', 'MOPLLAAG', 'ALEVEN', 'APLEZIER', 'ABRAND', 'CARAVAN']

# COMMAND ----------

# MAGIC %md Create the reduced dimension dataset to contain only the selected variables. 

# COMMAND ----------

red_caravan_df = caravan_df[most_imp_var]

# COMMAND ----------

# MAGIC %md Display the structure of the new dataset.

# COMMAND ----------

red_caravan_df.info()

# COMMAND ----------

# MAGIC %md ## 5. Explore Single Variables

# COMMAND ----------

# MAGIC %md ####  5.1 CARAVAN - Caravan Insurance Policy
# MAGIC Number of mobile home policies: 0 (no policy) or 1 (having policy)

# COMMAND ----------

grouped = red_caravan_df.groupby('CARAVAN')
grouped['CARAVAN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort_values(by='percentage', ascending =False)

# COMMAND ----------

# MAGIC %md 6% of all the zipcodes bought caravan insurance policy. 

# COMMAND ----------

caravan_plot = sns.barplot(x='CARAVAN', y='CARAVAN', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the above table. 94% of the zip codes did not purchase the insurance.

# COMMAND ----------

# MAGIC %md  ####  5.2 APERSAUT - Number of Car policies

# COMMAND ----------

grouped = red_caravan_df.groupby('APERSAUT')
grouped['APERSAUT'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The above result shows that most of the zipcodes are contentrated in the low levels of car insurance. 50% of them do not have any car insurance, 47% have 1-49 car insurances and 4% have 50-99 car insurances.  

# COMMAND ----------

caravan_plot = sns.barplot(x='APERSAUT', y='APERSAUT', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the above result. We can see clearly how all of the obseravtions fall into the low levels of car insurance policies. 

# COMMAND ----------

# MAGIC %md  ####  5.3 MINKGEM - Average Income

# COMMAND ----------

grouped = red_caravan_df.groupby('MINKGEM')
grouped['MINKGEM'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 33% of the zip codes have 24-36% average income and 32% of the zip codes have 37-49% average income. Our data mostly (about 64%) contains areas with 24-49% average income. 

# COMMAND ----------

caravan_plot = sns.barplot(x='MINKGEM', y='MINKGEM', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the average income result. We can see clearly how all of the observations fall into the middle levels average income. 

# COMMAND ----------

# MAGIC %md  ####  5.4 MKOOPKLA - Purchasing Power Class

# COMMAND ----------

grouped = red_caravan_df.groupby('MKOOPKLA')
grouped['MKOOPKLA'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 26% of the zip codes have 24-36% lower level education and 16% of the zip codes have 63-75% lower level education. Our data mostly contains areas with levels 3, 6 and 4 of this attribute. 

# COMMAND ----------

caravan_plot = sns.barplot(x='MKOOPKLA', y='MKOOPKLA', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the lower level education result. We can see clearly how most of the observations fall into the 3, 6 and 4 of of this attribute. 

# COMMAND ----------

# MAGIC %md  ####  5.5 MINKM30 - Income less than 30,000 

# COMMAND ----------

grouped = red_caravan_df.groupby('MINKM30')
grouped['MINKM30'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 22% of the zip codes have 0 percentage of income less than 30,000 and 37% have 11-36% of this type of income.

# COMMAND ----------

caravan_plot = sns.barplot(x='MINKM30', y='MINKM30', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the income less than 30,000 attribute. We can see clearly how most of the observations fall into the 0, 2 and 3 of of this attribute. 

# COMMAND ----------

# MAGIC %md  ####  5.6 AWAPART - Third Party Insurance 

# COMMAND ----------

grouped = red_caravan_df.groupby('AWAPART')
grouped['AWAPART'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 60% of the zip codes have 0 third party Insurance and 39.8% of the zipcodes have between 1-49 number of third party Insurance. 

# COMMAND ----------

caravan_plot = sns.barplot(x='AWAPART', y='AWAPART', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the third party Insurance attribute. We can see clearly how most of the observations most of the observations fall into the 0 and 1 level of this attribute.

# COMMAND ----------

# MAGIC %md #### 5.7 MHHUUR - Rented House

# COMMAND ----------

grouped = red_caravan_df.groupby('MHHUUR')
grouped['MHHUUR'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 17% of the zip codes have 0% of rented house and 12.7% of the zipcodes have 100% number of rented house.

# COMMAND ----------

caravan_plot = sns.barplot(x='MHHUUR', y='MHHUUR', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the rented house attribute. We can observe from the graph that 17% of the  observations fall into 0 level of this attribute which means that those zipcodes has 0% of rented house and rest of the levels are almost uniformly distributed.

# COMMAND ----------

# MAGIC %md #### 5.8 MOPLLAAG - Lower level education

# COMMAND ----------

grouped = red_caravan_df.groupby('MOPLLAAG')
grouped['MOPLLAAG'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 17.8% of the zip codes have 50-62% of lower level education people and 15% of the zipcodes have 63-75% number of lower level education people. We see that a high percentage of the people have lower level education in most of the zipcodes.

# COMMAND ----------

caravan_plot = sns.barplot(x='MOPLLAAG', y='MOPLLAAG', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the lower level education attribute. We can observe from the graph that most of the levels for this attribute are almost uniformly distributed and most of the zipcodes have people with lower level education.

# COMMAND ----------

# MAGIC %md #### 5.9 ALEVEN - Number of life insurances

# COMMAND ----------

grouped = red_caravan_df.groupby('ALEVEN')
grouped['ALEVEN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 95% of the zip codes dont have any Life Insurance policies and about 3.1% of the zipcodes have between 1-49 number of  Life Insurance policies .

# COMMAND ----------

caravan_plot = sns.barplot(x='ALEVEN', y='ALEVEN', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the Life Insurance policy attribute. We can observe from the graph that most of the zipcodes have people with no Life nsurance policy.

# COMMAND ----------

# MAGIC %md #### 5.10 APLEZIER - Number of boat policies

# COMMAND ----------

grouped = red_caravan_df.groupby('APLEZIER')
grouped['APLEZIER'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 99.5% of the zip codes dont have any Boat Insurance policies and about 0.5% of the zipcodes own between 1-99 number of Boat Insurance policies .

# COMMAND ----------

caravan_plot = sns.barplot(x='APLEZIER', y='APLEZIER', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the Boat Insurance policy attribute. We can observe from the graph that most of the zipcodes have people with no boat insurance policy.

# COMMAND ----------

# MAGIC %md #### 5.11 ABRAND - Number of fire policies

# COMMAND ----------

grouped = red_caravan_df.groupby('ABRAND')
grouped['ABRAND'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)}).sort('percentage', ascending =False)

# COMMAND ----------

# MAGIC %md The group by result shows that 45.5% of the zip codes dont have any Fire Insurance policies and 53% of zipcodes have between 1-49 any Fire Insurance policies.

# COMMAND ----------

caravan_plot = sns.barplot(x='ABRAND', y='ABRAND', data=red_caravan_df, estimator=lambda x: len(x) / len(red_caravan_df) * 100)
caravan_plot.set(ylabel="Percent")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md This graph visualizes the percentages of the Fire Insurance policy attribute. We can observe from the graph that almost 45% of the zipcodes have people with no fire insurance policy and rest 54% of the zipcodes have 1-49 number of fire insurance policies.

# COMMAND ----------

# MAGIC %md ## 6. Explore Multiple Variables

# COMMAND ----------

# MAGIC %md The purpose of the following EDA is to give a clear insight into customers having caravan insurance policy and how these customers are different from other customers. 

# COMMAND ----------

# MAGIC %md #### 6.1 Caravan vs Demographics

# COMMAND ----------

# MAGIC %md 1. Caravan, Average Income and Rented House

# COMMAND ----------

pd.pivot_table(red_caravan_df[['MHHUUR', 'CARAVAN','MINKGEM']], 
               values = 'MINKGEM',
               index  ='MHHUUR', 
               columns='CARAVAN', 
               aggfunc=np.mean)

# COMMAND ----------

# MAGIC %md The pivot table below shows the mean of average income for each combination of rented house and caravan insurance policy. The zip codes that bought caravan have average income between 24 to 62%. We can see that of all the zip codes that bought caravan the average income is higher for those that have lower percentage of rented house.

# COMMAND ----------

# MAGIC %md 2.Caravan and Purchasing Power Class

# COMMAND ----------

grouped = red_caravan_df.groupby(['CARAVAN', 'MKOOPKLA'])
grouped['CARAVAN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)})

# COMMAND ----------

# MAGIC %md The result above shows all of the combinations between the caravan two levels and the purchasing power class levels in terms of count and percentages. Most zipcodes fall into caravan 0 and purchising power class 3. We learn from these results that 25% of all the zip codes did not purshase caravan and have purchising power class of 24-36%. 

# COMMAND ----------

# MAGIC %md 3.Caravan vs Lower Level Education

# COMMAND ----------

grouped = red_caravan_df.groupby(['CARAVAN', 'MOPLLAAG'])
grouped['CARAVAN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)})

# COMMAND ----------

# MAGIC %md The above table result shows that among the zip codes that purchased caravan policy the first comes the ones with less lower level education.  

# COMMAND ----------

plt.gcf().clear()
grouped= red_caravan_df.groupby(['MOPLLAAG', 'CARAVAN'])
df=pd.DataFrame(grouped['CARAVAN'].agg({'count':np.size}))
sns.heatmap(df)
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md The above graph is the visualization of all the combinations of lower level education levels with those who purchased and who did not purchased caravan policy. The lighter the color the larger the number of zipcodes per that combination. The are with the most number of zipcodes correspond to the ones that did not buy caravan insurance and have lower level education of 50-62%. Also the next high number of zipcodes correspond to the ones that have lower level education of 50-75% and did not purchase caravan policy. 

# COMMAND ----------

# MAGIC %md #### 6.2 Caravan vs Other Insurances

# COMMAND ----------

# MAGIC %md 1.  Caravan and Car Insurance Policies

# COMMAND ----------

grouped = red_caravan_df.groupby(['CARAVAN','APERSAUT'])
grouped['CARAVAN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)})

# COMMAND ----------

# MAGIC %md The result above shows all of the combinations between the caravan two levels and the  car policies levels in terms of count and percentages. Most zipcodes fall into caravan 0 and car policies levels 0 and 1. We learn from these results that 90% of all the zip codes did not purshase caravan and have car policies between 0-49. 

# COMMAND ----------

# MAGIC %md 2.Caravan and Life insurances

# COMMAND ----------

grouped = red_caravan_df.groupby(['CARAVAN', 'ALEVEN'])
grouped['CARAVAN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)})

# COMMAND ----------

# MAGIC %md The above result shows that out of the zipcodes that fall into caravan = 0, 94% of the zipcodes do not have any life insurance policy. Similar pattern is obersved for zipcodes that fall under caravan = 1.

# COMMAND ----------

df = red_caravan_df.groupby(['CARAVAN', 'ALEVEN'])['CARAVAN'].count().unstack('ALEVEN').fillna(0)
df=pd.DataFrame(df)
df.plot(kind='bar',stacked=True,title="Caravan vs #life insurances")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md The above chart shows that irrespective of the zipcode being 0 and 1, most of the zipcodes do not have any life insurance policies

# COMMAND ----------

# MAGIC %md 3.Caravan and Fire Insurance Policy

# COMMAND ----------

grouped = red_caravan_df.groupby(['CARAVAN', 'ABRAND'])
grouped['CARAVAN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)})

# COMMAND ----------

# MAGIC %md The result above shows all of the combinations between the caravan two levels and the fire policies levels in terms of count and percentages. Most zipcodes  that fall into caravan 0  belong car policy levels 0 and 1. We learn from these results that 92% of all the zip codes did not purshase caravan and have fire policies between 0-49. 

# COMMAND ----------

df1 = red_caravan_df.groupby(['CARAVAN', 'ABRAND'])['CARAVAN'].count().unstack('ABRAND').fillna(0)
my_df1=pd.DataFrame(df1)
my_df1.plot(kind='bar',stacked=True,title="Caravan vs #fire insurance policies")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md The bove chart shows an interesting analysis compared to the previous chart. Irrespective whether caravan is 0 or 1, we see that there is a good proportion of zipcodes who have fire insurance. 

# COMMAND ----------

# MAGIC %md 4.Caravan and Third Party Insurance

# COMMAND ----------

grouped = red_caravan_df.groupby(['CARAVAN', 'AWAPART'])
grouped['CARAVAN'].agg({'count':np.size,
                  'percentage': lambda x: len(x) / len(red_caravan_df)})

# COMMAND ----------

# MAGIC %md The above cross tab shows that 39% of the zipcodes, who fall under carvan = 0 category, have one or more third party insurances. While for zipcodes falling under carvan = 1, almost 58% of the them have one or more third party insurances.

# COMMAND ----------

df1 = red_caravan_df.groupby(['CARAVAN', 'AWAPART'])['CARAVAN'].count().unstack('AWAPART').fillna(0)
my_df1=pd.DataFrame(df1)
my_df1.plot(kind='bar',stacked=True,title= "CARAVAN vs private third party policies")
display()
plt.gcf().clear()

# COMMAND ----------

# MAGIC %md The above chart shows similar trend as fire policies. Almost 39% of the zipcodes, who fall under caravan = 0 category have private third party insurance and 58% for those with carvan = 1.

# COMMAND ----------

# MAGIC %md ### Conclusion

# COMMAND ----------

# MAGIC %md We started the analysis with the 87 variables that were available initially. We use correlation values at the first step to filter out the significant variables required for the analyses. The selected variables were uncorrelated with each other. Random forests technique was used to identify any other insignificant variables for prediction. This resulted in elimination of other insignificant variables. Some of the important factors using the random forest analyses were #fire policies, #car policies, #life insurances, #boat policies, other third-party insurances, income level and purchasing power.
# MAGIC 
# MAGIC Then we further drilled down to see how the final shortlisted variables vary with the dependent variable. We used bi-variate analyses including visualizations and frequency reports to see how these independent variables varied with the dependent variable.