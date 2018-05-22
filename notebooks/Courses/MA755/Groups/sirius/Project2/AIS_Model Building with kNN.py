# Databricks notebook source
# MAGIC %md #AIS Model Building with kNN

# COMMAND ----------

# MAGIC %md # 1. Introduction

# COMMAND ----------

# MAGIC %md The objective of this notebook is to use a pipeline that transforms the data step by step and performs the `kNN Regression` estimator. Each step of the pipeline is explained how additional features are appended to the data and then aggregated at the date level. The new features include lags, window features, datetime features, location features and status features. In this notebook, the daily oil price change has been added as an external feature. Each feature is generated using different classes. The last step of the pipeline applies an estimator to variables created from the previous steps for predicting the `Average` shipping fares. We performed grid search to find the best hyperparameters for kNN. Then, we build the model using the train data and make predictions to evaluate our model performance on the train and test dataset. The model performance is measured using the `RMSE` and the `Explained Variance`.

# COMMAND ----------

# MAGIC %md Below are the analyses steps that I specifically worked on during the model building phase:
# MAGIC 
# MAGIC __Defining `AddLocationAndAggregate` class:__
# MAGIC 
# MAGIC - Researched on geopy, reverse_geocode packages to find the location using a set of coordinates. Added location columns `City`, `Country` and `country_code` using latitude and longitude coordinates 
# MAGIC - Identified the outlier longitudinal values present in the dataset using visualizations and capped the outlier longitudinal values to 180
# MAGIC -  The number of ships at each location was a key predictor for shipping fares. 
# MAGIC      - In order to extract the above information, I binarized the `Status` and the `Country` variables before aggregation
# MAGIC - Next step was to get the data available at the unique date level 
# MAGIC      - Grouped the data by `Date` and `Average` and aggregated other variables by the sum
# MAGIC - Aggregated output of the `Country` column tells the number of ships at that location
# MAGIC - Separately, grouped `Date` and aggregate the `dwt` variable by the mean
# MAGIC - Merge the two aggregated outputs to prepare the data at the unique day level for prediction 
# MAGIC      - Concatenated the last two steps and return the concatenated dataframe as output
# MAGIC      
# MAGIC - Applied the same set of steps for both train and test data
# MAGIC 
# MAGIC 
# MAGIC __Defining `ExternalFeatures` class__:
# MAGIC - Researched on external features affecting the shipping fares. Checked online resources/research papers to identify the external factors affecting the shipping fares. Included oil prices as an external predictor from an external data source
# MAGIC 
# MAGIC - The data source provides the daily price of the crude stream traded at Cushing, Oklahoma, which is used as a benchmark in oil pricing (https://fred.stlouisfed.org/series/DCOILWTICO)
# MAGIC 
# MAGIC - Downloaded oil prices for the train and test data time periods separately and appended them to the aggregated data at unique date level
# MAGIC 
# MAGIC 
# MAGIC __Defining `ResetIndex` class__:
# MAGIC 
# MAGIC - Before applying the kNN estimator in the pipeline, this class resets the index and removes the index column
# MAGIC 
# MAGIC 
# MAGIC __Grid Search__:
# MAGIC 
# MAGIC - After the first model iteration with kNN, used Grid-Search to build a model on each parameter combination possible. The function iterates through every parameter combination and stores a model for each combination
# MAGIC - Parameters varied: k nearest neighbors, number of cross-folds and number of jobs in parallel
# MAGIC 
# MAGIC As part of a group project, all the three members worked collaboratively understanding each section of the notebook.

# COMMAND ----------

# MAGIC %md #Summary of the Analyses steps
# MAGIC 
# MAGIC __1. Introduction__
# MAGIC 
# MAGIC - Objective and purpose of this notebook
# MAGIC 
# MAGIC __2. Set up__
# MAGIC 
# MAGIC Import other notebooks to make all the custom classes and functions available
# MAGIC - Import Data Preparation notebook 
# MAGIC - Import Exploratory Data Analysis notebook
# MAGIC - Import Data Setup and Feature Engineering notebook
# MAGIC 
# MAGIC __3. Explain Pipeline Steps__
# MAGIC 
# MAGIC Create objects and mappers using classes defined in the imported notebooks 
# MAGIC - 3.1 Add location and External Features object (_Tauseef_)
# MAGIC     - Create an object of `AddLocationAndAggregate` class for the train data. Derive location of ships using geolocation coordinates and aggregates the data 
# MAGIC     - Create an object of `ExternalFeatures` class for the train and test data. Add percentage change in oil prices
# MAGIC     
# MAGIC - 3.2 Mappers and Feature Union
# MAGIC      - Lag mapper
# MAGIC         - Create a `lag_mapper` that uses the `Lags` class to generate lags from the `Average` column. Lags of the shipping fares to be as a model input
# MAGIC      - DateTimeGenerator mapper
# MAGIC         - Create a `datetime_mapper` that uses the `DatetimeGenerator` class to generate month, day and year columns from the `Date` column
# MAGIC      - MyX mapper
# MAGIC         - Mapping train and test data features to make sure both have the identical features
# MAGIC      - Feature union
# MAGIC         - Merges the output of `Lags`, `DatetimeGenerator` and `MyX` classes 
# MAGIC         
# MAGIC - 3.3 MakeDataframe object
# MAGIC 
# MAGIC     - Create an object of `MakeDataframe` class which converts the numpy array output of `feature union` to a pandas dataframe and names the columns by the list given in`col_names` parameter
# MAGIC     
# MAGIC - 3.4 WindowFeature object
# MAGIC 
# MAGIC     - This class uses the output of `MakeDataframe` class and performs rolling window calculations
# MAGIC     
# MAGIC - 3.5 Remove NA's object
# MAGIC 
# MAGIC     - To drop the rows with the NA values created from the `Lags` class and the `WindowFeature` class
# MAGIC     
# MAGIC - 3.6 Reset Index (_Tauseef_)
# MAGIC 
# MAGIC     - This class resets the index and removes the index column from the dataset
# MAGIC     
# MAGIC __4. Pipeline__
# MAGIC 
# MAGIC - Define the pipeline
# MAGIC 
# MAGIC     - Sequentially applying a list of transforms and a final estimator
# MAGIC     - List of transforms: add location and aggregate, add oil data, feature union, convert to dataframe, window feature, remove NA, reset index, standardization
# MAGIC     - Estimator: kNN
# MAGIC     
# MAGIC - 4.1 Prediction train part
# MAGIC     - After fitting the train data into the pipeline, measure model accuracy by predicting shipping fares on the training dataset 
# MAGIC     
# MAGIC - 4.2 Prediction test part
# MAGIC     - Forecast shipping fares on the test data
# MAGIC     
# MAGIC __5. Grid Search__ (_Tauseef_)
# MAGIC 
# MAGIC - Configure the model with the optimal set of hyperparameters to find the best model fit
# MAGIC 
# MAGIC __6. Conclusion__
# MAGIC 
# MAGIC  - Next Steps

# COMMAND ----------

# MAGIC %md # 2. Setup 

# COMMAND ----------

# MAGIC %md In this section we import the Data Setup and Feature Engineering and Data Preparation notebooks to make the all the data and classes available from the connected notebooks reading data and class definition.

# COMMAND ----------

# MAGIC %md Import the Data Preparation Notebook.

# COMMAND ----------

# MAGIC %run "/Courses/MA755/Groups/sirius/Project2/Data Preparation" 

# COMMAND ----------

# MAGIC %md Import the Data Setup and Feature Engineering Notebook.

# COMMAND ----------

# MAGIC %run "/Courses/MA755/Groups/sirius/Project2/Data Setup and Feature Engineering"

# COMMAND ----------

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# COMMAND ----------

# MAGIC %md # 3. Explain Pipeline Steps

# COMMAND ----------

# MAGIC %md Define the number of lags that will be used for the `Average` column, and the window width that will be used for the window statistics.

# COMMAND ----------

lag_no = 5
window_width = 2

# COMMAND ----------

# MAGIC %md ## 3.1 `AddLocationAndAggregate` and `ExternalFeatures`

# COMMAND ----------

# MAGIC %md ##### `AddLocationAndAggregate` object for Train Data

# COMMAND ----------

# MAGIC %md Create an object of `AddLocationAndAggregate` class and check the output of its transform method. The input of transform method is the train data.

# COMMAND ----------

prepare_train = AddLocationAndAggregate()
new_train = prepare_train.fit_transform(X_train,y_train)
new_train.shape, new_train.head(4)

# COMMAND ----------

# MAGIC %md The output of the transform method of `AddLocationAndAggregate` class is a dataframe with 999 rows and 181 columns. Each row corresponds to a unique date of the train data. The columns are the date, the average value, three columns for each status, 175 columns for each country and the dead weight. The average value corresponds to each date while the three variables related to status indicate how many ships are in that date in each status level. The 175 country variables indicate how many ships are in that date in each particular country. The dead weight is the mean of the dead weight of all the ships for that date.

# COMMAND ----------

# MAGIC %md  ##### `ExternalFeatures` object for Train Data

# COMMAND ----------

# MAGIC %md Create an object of `ExternalFeatures` class for the train data and check the output of its transform method. The input is the output of the `AddLocationAndAggregate` class.

# COMMAND ----------

add_oil_train = ExternalFeatures()
new_train = add_oil_train.fit_transform(new_train)
new_train.shape, new_train.head(4)

# COMMAND ----------

# MAGIC %md The output of the `ExternalFeatures` transform method is a new dataframe that contains all of the columns of the `AddLocationAndAggregate` and the column with the crude oil price percentage change. This new dataframe has 999 rows and 182 columns.

# COMMAND ----------

# MAGIC %md ##### `AddLocationAndAggregate` object for Test Data

# COMMAND ----------

# MAGIC %md Create an object of `AddLocationAndAggregate` class and check the output of its transform method. The input of transform method is the test data.

# COMMAND ----------

prepare_test = AddLocationAndAggregate()
new_test = prepare_test.fit_transform(X_test)
new_test.shape, new_test.head(4)

# COMMAND ----------

# MAGIC %md The output of the transform method of `AddLocationAndAggregate` class is a dataframe with 271 rows and 174 columns. Each row corresponds to a unique date of the test data. The columns are the date, the average value, three columns for each status, 168 columns for each country and the dead weight. The average value corresponds to each date while the three variables related to status indicate how many ships are in that date in each status level. The 168 country variables indicate how many ships are in that date in each particular country. The dead weight is the mean of the dead weight of all the ships for that date.

# COMMAND ----------

# MAGIC %md  #### `ExternalFeatures` object for Test Data

# COMMAND ----------

# MAGIC %md Generate an object of `ExternalFeatures` class for the test data and check the output of its transform method. The input is the output of the `AddLocationAndAggregate` class.

# COMMAND ----------

add_oil_test = ExternalFeatures()
new_test = add_oil_test.fit_transform(new_test)
new_test.shape, new_test.head(4)

# COMMAND ----------

# MAGIC %md The output of the `ExternalFeatures` transform method is a new dataframe that contains all of the columns of the `AddLocationAndAggregate` and the column with the crude oil price percentage change. This new dataframe has 271 rows and 175 columns.

# COMMAND ----------

# MAGIC %md ## 3.2 Mappers and `FeatureUnion`

# COMMAND ----------

# MAGIC %md ##### `Lags` Mapper

# COMMAND ----------

# MAGIC %md Create a `lag_mapper` that uses the `Lags` class to generate lags from the `Average` column, and check the output of the transform method. The input is the aggregated data in `Date` level.

# COMMAND ----------

lag_mapper =  DataFrameMapper([('Average', Lags(lag_no))],                            
                                 df_out=True)
lag_mapper.fit_transform(new_train).head(4)

# COMMAND ----------

# MAGIC %md The output above is a dataframe containing the lags generated from the `Average` variable. There are five lags generated. There are NA values because the past values at those points do not exist.

# COMMAND ----------

# MAGIC %md ##### `DatetimeGenerator` Mapper

# COMMAND ----------

# MAGIC %md Create a `datetime_mapper` that uses the `DatetimeGenerator` class to generate month, day and year columns from the `Date` column, and check the output of the transform method. The input is the aggregated data in `Date` level. The output is the aggregated data at `Date` level.

# COMMAND ----------

date_list=['Date']
date_mapper = DataFrameMapper(gen_features(columns=date_list, 
                                             classes=[{'class': DatetimeGenerator}
                                                     ]), df_out=True)
date_mapper.fit_transform(new_train).head()

# COMMAND ----------

# MAGIC %md The output of the `datetime_mapper` is a dataframe containing the month, day and year columns which are aggregated at date level.

# COMMAND ----------

# MAGIC %md ##### `MyX` mapper

# COMMAND ----------

# MAGIC %md The train and test data should have the same number of columns in order for the model to be used for prediction. Since there are some countries in the train data that do not appear in the test data we remove those columns from train. The `list_of_cols` contains the column names that appear in both train and test. 

# COMMAND ----------

list_of_cols = [col for col in new_train.columns if col in new_test.columns]

# COMMAND ----------

# MAGIC %md Create a `myx_mapper` that uses the `MyX` class to preserve the columns in `cols_list` unchanged. The input is the aggregated data at `Date` level. 

# COMMAND ----------

cols_list = [col for col in list_of_cols if col not in ['Average','Date']]
myx_mapper = DataFrameMapper(gen_features(columns=cols_list,
                               classes=[{'class': MyX}]),
                          df_out=True)
myx_mapper.fit_transform(new_train).head()

# COMMAND ----------

# MAGIC %md The output is the same as the input. It returns the columns given in `cols_list` the way they are. 

# COMMAND ----------

# MAGIC %md  ##### `FeatureUnion`

# COMMAND ----------

# MAGIC %md Create a `feature_union` that merges the output of `Lags`, `DatetimeGenerator` and `MyX` classes. The input is the aggregated data at `Date` level. The output is a numpy array.

# COMMAND ----------

feature_union = sklearn.pipeline.FeatureUnion([('lag features',    lag_mapper),
                                              ('date time features',    date_mapper),
                                               ('myx', myx_mapper)
                                              ])    
feature_union_output = feature_union.fit_transform(new_train)
feature_union_output, feature_union_output.shape

# COMMAND ----------

# MAGIC %md The output of `feature_union` is a numpy array which contains the concatenated output of `lag_mapper`, `datetime_mapper` and `myx_mapper`.

# COMMAND ----------

# MAGIC %md ## 3.3 `MakeDataframe` Object

# COMMAND ----------

# MAGIC %md Generate an object of `MakeDataframe` class and check the output of its transform method. The input is the numpy array output of the `feature_union`.

# COMMAND ----------

col_names=['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'month', 'day', 'year'] + cols_list
make_dataframe = MakeDataframe(col_names)
make_dataframe_output = make_dataframe.fit_transform(feature_union_output)
make_dataframe_output.shape, make_dataframe_output.head(4)

# COMMAND ----------

# MAGIC %md The output of the `MakeDataframe` object is a pandas dataframe with column names as given in `col_names` parameter.

# COMMAND ----------

# MAGIC %md ## 3.4 `WindowFeature` Object

# COMMAND ----------

# MAGIC %md Generate an object of `WindowFeature` class and check the output of its transform method. The input is the output of `MakeDataframe` class.

# COMMAND ----------

window_generator = WindowFeature(window_width)
window_generator_output = window_generator.fit_transform(make_dataframe_output)
window_generator_output.shape, window_generator_output.head(4)

# COMMAND ----------

# MAGIC %md The output of the above `WindowFeature` object is a dataframe containing the windows generated from the lag variables. There are five window features generated. There are NA values because the past values at those points do not exist.

# COMMAND ----------

# MAGIC %md ## 3.5 `RemoveNa` Object

# COMMAND ----------

# MAGIC %md Generate an object of `RemoveNa` class and check the output of its transform method. The input is the output of `WindowFeature` class.

# COMMAND ----------

remove_na = RemoveNa()
remove_na_output=remove_na.fit_transform(window_generator_output)
remove_na_output.shape, remove_na_output.head(4)

# COMMAND ----------

# MAGIC %md The output of the above `RemoveNa` object is a dataframe that does not contain any row with NA values. But, the index starts from 6, so we need to reset the index in the nest step using the `ResetIndex` class.

# COMMAND ----------

# MAGIC %md ## 3.6 `ResetIndex` Object

# COMMAND ----------

# MAGIC %md Generate an object of `ResetIndex` class and check the output of its transform method. The input is the output of the RemoveNa class. This class resets and removes the index column to prepare the input for the final estimator.

# COMMAND ----------

reset_index = ResetIndex()
X_train_final = reset_index.fit_transform(remove_na_output)
X_train_final.shape, X_train_final.head(4)

# COMMAND ----------

# MAGIC %md The output of the above `ResetIndex` object is a final version of the train data. It is a dataframe with 993 rows and 184 columns. The index is reset to start from zero. This data will be the input of the standard scaler step in the following pipeline. 

# COMMAND ----------

# MAGIC %md ## 4. Pipeline

# COMMAND ----------

# MAGIC %md In this section, we train the model in the pipeline and evaluate it.

# COMMAND ----------

# MAGIC %md The pipeline below gets the train data as input and performs the above explained steps. Then, it scales the data using the `sklearn StandardScaler` and trains the model with `sklearn KNeighborsRegressor`.

# COMMAND ----------

pipeline_1 = sklearn.pipeline.Pipeline([('add location and aggregate', AddLocationAndAggregate()),
                                        ('add oil data', ExternalFeatures()),
                                        ('feature union', feature_union),
                                        ('make dataframe', MakeDataframe(col_names)),
                                        ('window features', WindowFeature(window_width)),
                                        ('remove na', RemoveNa()),
                                        ('reset index', ResetIndex()),
                                        ('scaler', sklearn.preprocessing.StandardScaler()),
                                       ('KNN regression', KNeighborsRegressor(n_neighbors=30))])

# COMMAND ----------

# MAGIC %md #### 4.1 Prediction for train data

# COMMAND ----------

# MAGIC %md Fit the data into pipeline and make predictions using the train set. Save these predictions to `y_pred_train`.

# COMMAND ----------

# Make predictions using the train set
c = pipeline_1.fit(X_train,y_train_scaled)
y_pred = c.predict(X_train)

# COMMAND ----------

y_pred[:10]

# COMMAND ----------

# MAGIC %md Evaluate the performance of this model on train set by checking the explained variance, RMSE and the coefficients of the model. 

# COMMAND ----------

# Explained variance score: 1 is perfect prediction
Variance = r2_score(y_train_scaled, y_pred)
print('Variance score: ', Variance)
# Print the RMSE
print('RMSE: ', math.sqrt(mean_squared_error(y_train_scaled, y_pred)))
# The coefficients 
Coefficients = c.named_steps['KNN regression'].score
print('Coefficients: \n', Coefficients)

# COMMAND ----------

# MAGIC %md The explained variance score is the R square value. It shows that 84.4 % of the changes in Average value can be explained by these predictors. This is a high R square but this is the evaluation of the model in the train data so, it is expected to be high. The RMSE of the model is 0.3947. The values in the list are the coefficients.  

# COMMAND ----------

# MAGIC %md #### 4.2 Prediction for test data

# COMMAND ----------

# Make predictions using the test set
c = pipeline_1.fit(X_train,y_train_scaled)
y_pred2 = c.predict(X_test)
y_pred2[:10]

# COMMAND ----------

# MAGIC %md Evaluate the performance of this model on  test set by checking the explained variance and the RMSE of the model. 

# COMMAND ----------

# Explained variance score: 1 is perfect prediction
Variance = r2_score(y_test_scaled, y_pred2)
print('Variance score:', Variance)
# print the RMSE
print('RMSE: ', math.sqrt(mean_squared_error(y_test_scaled, y_pred2)))

# COMMAND ----------

# MAGIC %md The RMSE of the test data is very high compared to the train RMSE. This might be caused from overfitting the data. The test R square value is also very low compared to the train R square. This means that this model explains 19% of the change in `Average` price.

# COMMAND ----------

# MAGIC %md #5. Grid Search 
# MAGIC 
# MAGIC Parameters that are not directly learnt within estimators can be set by searching a parameter space for the best. The method `best_score_` finds the best score observed during the optimization procedure.

# COMMAND ----------

# MAGIC %md ##### Create a `KNeighborsRegressor` object.

# COMMAND ----------

knn = KNeighborsRegressor()

# COMMAND ----------

# MAGIC %md Define the parameter values that should be searched. Defining range of k nearest neighbours

# COMMAND ----------

k_range = list(range(1, 31))

# COMMAND ----------

# MAGIC %md Define the `param_grid` dictionary.

# COMMAND ----------

param_grid = dict(n_neighbors=k_range)
print(param_grid)

# COMMAND ----------

# MAGIC %md Perform the grid search using the previously defined parameters and 10 folds for cross validation.

# COMMAND ----------

grid = GridSearchCV( estimator=knn, 
                         param_grid = param_grid,
                         cv       = 10,
                         n_jobs =4)

grid

# COMMAND ----------

# MAGIC %md Scale the train data using `StandardScaler`

# COMMAND ----------

scaled_x_train = sklearn.preprocessing.StandardScaler().fit_transform(X_train_final)

# COMMAND ----------

# MAGIC %md Fit the train data to the `grid`

# COMMAND ----------

grid.fit(scaled_x_train, y_train_scaled)

# COMMAND ----------

# MAGIC %md Print the `grid_scores`

# COMMAND ----------

grid.grid_scores_

# COMMAND ----------

# MAGIC %md Print the best score, the best parameter and the best estimator

# COMMAND ----------

# Single best score achieved across all params (k)
print(grid.best_score_)

# Dictionary containing the parameters (k) used to generate that score
print(grid.best_params_)

# Actual model object fit with those best parameters
print(grid.best_estimator_)

# COMMAND ----------

# MAGIC %md Print the parameters of corresponding to the best score 

# COMMAND ----------

print('Parameters')
print(grid.grid_scores_[29].parameters)
# Array of 10 accuracy scores during 10-fold cv using the parameters
print('CV Validation Score')
print(grid.grid_scores_[29].cv_validation_scores)
# Mean of the 10 scores
print('Mean Validation Score')
print(grid.grid_scores_[29].mean_validation_score)

# COMMAND ----------

# MAGIC %md # 6. Conclusion

# COMMAND ----------

# MAGIC %md In this notebook we defined a pipeline that processes and scales the data and builds a linear regression model on it. The pipeline uses all of the classes defined in the previous notebook to process the data. This phase happens in the first seven steps where new features are added including the oil price, the data is aggregated at date level, lag and window features are created and NA values are removed. Each of the steps that transform the data into the desired form are explained in the first part of the notebook. Before the estimator, we scale the data using the Standard Scaler.  Next, we perform the grid search using a range of 1 to 30 for k values and 10 folds for cross validation. Using the best parameters from the grid search, we fit the `KNN regression` model to the scaled train data. When we evaluated the model performance in train set, the explained variance appeared to be 0.8842 while the RMSE was 0.3947. This model has a high R square and high RMSE for the train model in this dataset. The most important fact is how this model performs on unseen data. So, we tested the model performance on the test dataset and got a variance score of 0.19 and RMSE value of 0.899. Higher RMSE and lower variance might indicate that this model does not perform well. 
# MAGIC 
# MAGIC __Next steps:__
# MAGIC 
# MAGIC - Try new data transformations to improve the model fit. 
# MAGIC - Analyze external factors including gdp, inflation and demand far raw materials
# MAGIC - Use dimension reduction to find the combine the ship locations with similar attributes
# MAGIC - Fit other modeling techniques including Linear Regression and Random Forest