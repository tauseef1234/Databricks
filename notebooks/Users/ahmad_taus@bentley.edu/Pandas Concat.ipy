# Databricks notebook source
# MAGIC %md # `pandas` - Concatenation

# COMMAND ----------

# MAGIC %md __Contents:__
# MAGIC 
# MAGIC 1.  Concatenating objects
# MAGIC 2.  Set logic on other axes
# MAGIC 3.  Concatenating using append
# MAGIC 4.  Ignoring indexes on the concatenation axis
# MAGIC 5.  Concatenating 

# COMMAND ----------

# MAGIC %md Related/useful documentation:
# MAGIC - https://pandas.pydata.org/pandas-docs/stable/merging.html
# MAGIC - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html
# MAGIC - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html

# COMMAND ----------

# MAGIC %md ### Load libraries

# COMMAND ----------

import pandas  as pd
import numpy   as np
(pd.__version__,
 np.__version__
)

# COMMAND ----------

# MAGIC %md __Concatenation__
# MAGIC 
# MAGIC `concat` function - The concat function does all of the heavy lifting of performing concatenation operations along an axis while performing optional set logic (union or intersection) of the indexes (if any) on the other axes.
# MAGIC 
# MAGIC https://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html

# COMMAND ----------

# MAGIC %md __1. Concatenating Objects__
# MAGIC 
# MAGIC pandas.concat takes a list or dict of homogeneously-typed objects and concatenates them with some configurable handling of “what to do with the other axes”:

# COMMAND ----------

# MAGIC %md __Example__
# MAGIC 
# MAGIC Defining sample data panda dataframes df1 and df2

# COMMAND ----------

 df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
 df1

# COMMAND ----------

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])
df2

# COMMAND ----------

frames = [df1, df2]

# COMMAND ----------

# MAGIC %md Using concat function to concatenate two dataframes df1 and df2. The concatenated dataframed in stored in `result`.

# COMMAND ----------

result = pd.concat(frames)
result

# COMMAND ----------

# MAGIC %md Suppose we wanted to associate specific keys with each of the pieces of the chopped up DataFrame. We can use the keys argument:

# COMMAND ----------

result = pd.concat(frames, keys=['x', 'y']).index
result

# COMMAND ----------

# MAGIC %md The resulting object's index has a hierarchical index. We can select each chunk by key:

# COMMAND ----------

result.loc['y']

# COMMAND ----------

# MAGIC %md __2. Set logic on other axes__:
# MAGIC 
# MAGIC While appending multiple data frames, you have a choice how to handle other axes. This can be done in three ways:
# MAGIC 
# MAGIC 1. `join = 'outer'` which takes sorted union of all, zero information loss
# MAGIC 2. `join = 'inner'` which takes the intersection
# MAGIC 3. `join_axes` Use a specific index or indexes

# COMMAND ----------

# MAGIC %md `join = 'outer'`

# COMMAND ----------

df3 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])
df3

# COMMAND ----------

df1

# COMMAND ----------

result = pd.concat([df1, df3], axis=1)
result

# COMMAND ----------

# MAGIC %md `join` = 'inner'

# COMMAND ----------

result = pd.concat([df1, df3], axis=1, join='inner')
result

# COMMAND ----------

# MAGIC %md `join_axes`

# COMMAND ----------

result = pd.concat([df1, df3], axis=1, join_axes=[df1.index])
result

# COMMAND ----------

# MAGIC %md __3. Concatenating using append__
# MAGIC 
# MAGIC A useful shortcut to concat are the append instance methods on Series and DataFrame. These methods actually predated concat. They concatenate along axis=0, namely the index:

# COMMAND ----------

result = df1.append(df2)
result

# COMMAND ----------

# MAGIC %md `append` may take multiple objects to concatenate:

# COMMAND ----------

result = df1.append([df2, df3])
result

# COMMAND ----------

# MAGIC %md __4. Ignoring indexes on the concatenation axis__
# MAGIC 
# MAGIC For DataFrames which don’t have a meaningful index, you may wish to append them and ignore the fact that they may have overlapping indexes. We can set `ignore_index` = `True`. The same argument works in a similar way with `DataFrame.append`

# COMMAND ----------

result = pd.concat([df1, df3], ignore_index=True)
result

# COMMAND ----------

# MAGIC %md __5. Concatenating with mixed ndims__
# MAGIC 
# MAGIC We can also concatenate a mix of Series and DataFrames. The Series gets transformed to DataFrames with the column name as the name of the Series.

# COMMAND ----------

s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')
s1

# COMMAND ----------

result = pd.concat([df1, s1], axis=1)
result

# COMMAND ----------

# MAGIC %md ##Example of concact using the iris dataset

# COMMAND ----------

iris = pd.read_csv('/dbfs/mnt/datalab-datasets/file-samples/iris.csv')
iris

# COMMAND ----------

iris.shape

# COMMAND ----------

# MAGIC %md Split the dataset into two dataframes of different sizes

# COMMAND ----------

from sklearn.model_selection import train_test_split
iris_df1,iris_df2= train_test_split(iris,test_size=0.4,train_size=0.6)

# COMMAND ----------

# MAGIC %md Now we will try applying the functions on the two subsets (`iris_df1`,`iris_df2`) of the iris dataset

# COMMAND ----------

frame = [iris_df1, iris_df2]

# COMMAND ----------

iris_concat1 = pd.concat(frame, keys=['x', 'y'])
iris_concat1

# COMMAND ----------

iris_concat2= pd.concat([iris_df1, iris_df2])
iris_concat2.shape

# COMMAND ----------

iris_df1

# COMMAND ----------

iris_append = iris_df1.append(iris_df2)
iris_append.shape

# COMMAND ----------

iris

# COMMAND ----------

# MAGIC %md __The End__