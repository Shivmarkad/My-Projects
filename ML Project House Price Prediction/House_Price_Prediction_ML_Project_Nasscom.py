# Import the required modules and load the dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/house-prices.csv')

df.head()

# Get the information on DataFrame.

df.info()

# Check if there are any NULL values.

df.isnull().sum()


# Check categorical attributes

df1 = df.select_dtypes(include = 'object')

df1.head()

# Boxplot for 'mainroad' vs 'price'

plt.figure(figsize = (18,9))

plt.title('Box plot between mainroad and price')

sns.boxplot(x = 'mainroad',y = 'price',data = df)

plt.show()

# Boxplot for 'guestroom' vs 'price'

plt.figure(figsize = (18,9))

plt.title('Box plot between guestroom and price')

sns.boxplot(x = 'guestroom',y = 'price',data = df)

plt.show()

# Boxplot for 'basement' vs 'price'

plt.figure(figsize = (18,9))

plt.title('Box plot between basement and price')

sns.boxplot(x = 'basement',y = 'price',data = df)

plt.show()

# Boxplot for 'hotwaterheating' vs 'price'

plt.figure(figsize = (18,9))

plt.title('Box plot between hotwaterheating and price')

sns.boxplot(x = 'hotwaterheating',y = 'price',data = df)

plt.show()

# Boxplot for 'airconditioning' vs 'price'

plt.figure(figsize = (18,9))

plt.title('Box plot between airconditioning and price')

sns.boxplot(x = 'airconditioning',y = 'price',data = df)

plt.show()

# Boxplot for 'prefarea' vs 'price'

plt.figure(figsize = (18,9))

plt.title('Box plot between prefarea and price')

sns.boxplot(x = 'prefarea',y = 'price',data = df)

plt.show()

# Boxplot for 'furnishingstatus' vs 'price'

plt.figure(figsize = (18,9))

plt.title('Box plot between furnishingstatus and price')

sns.boxplot(x = 'furnishingstatus',y = 'price',data = df)

plt.show()

# Create scatter plot with 'area' on X-axis and 'price' on Y-axis

plt.figure(figsize = (12,9))

plt.title('Scatter plot between area and price')

plt.scatter('area', 'price',data = df)

plt.xlabel('area')

plt.ylabel('price')

plt.show()

# Create scatter plot with 'bedrooms' on X-axis and 'price' on Y-axis

plt.figure(figsize = (12,9))

plt.title('Scatter plot between bedroom and price')

plt.scatter('bedrooms', 'price',data = df)

plt.xlabel('bedrooms')

plt.ylabel('price')

plt.show()

# Create scatter plot with 'bathrooms' on X-axis and 'price' on Y-axis

plt.figure(figsize = (12,9))

plt.title('Scatter plot between bathrooms and price')

plt.scatter('bathrooms', 'price',data = df)

plt.xlabel('stories')

plt.ylabel('price')

plt.show()

# Create scatter plot with 'stories' on X-axis and 'price' on Y-axis

plt.figure(figsize = (12,9))

plt.title('Scatter plot between stories and price')

plt.scatter('stories', 'price',data = df)

plt.xlabel('stories')

plt.ylabel('price')

plt.show()

# Create a normal distribution curve for the 'price'.

# Create a probablity density function for plotting the normal distribution


def probability_density(series):

  const = 1/(series.std()*np.sqrt(2*np.pi))

  power_of_e = -(series- series.mean())**2/(2*series.var()**2)

  probability_density = const * np.exp(power_of_e)

  return probability_density


#  the normal distribution curve using plt.scatter() 


series = df['price']

prob_d = probability_density(series)

plt.figure(figsize= (19,8))

plt.scatter(series,prob_d)

plt.title('Normal distribution for price of diamonds.')

plt.axvline(x = series.mean(),label = 'Mean')

plt.xlabel('Price of Diamonds')

plt.legend()

plt.show()


# Replace yes with 1 and no with 0 for all the values in features 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea' using map() function.

num_dict = {'yes':1, 'no':0}        # defining a dictionary to replace yes and no with 1 and 0


def num_map(series):                #defining a function to map dictionary in series

  return series.map(num_dict)

# replacing yes and no in datafram with 1 and 0

df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']] = df[['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']].apply(num_map, axis = 1)

# Print dataframe

print('DataFrame after replacing yes and no with 1 and 0 :\n',df.head())

# Getting counts of each value in furnishingstatus

df['furnishingstatus'].value_counts()

# Perform one hot encoding for furnishingstatus feature.

df_dummies = pd.get_dummies(df['furnishingstatus'], dtype = int)

print(df_dummies)

# Drop 'furnishingstatus' feature

df.drop(columns= 'furnishingstatus', inplace = True)

df = pd.concat([df, df_dummies], axis = 1)

# Print dataframe 

df.head()         # Display of first five rows of dataframe

# Split the 'df' Dataframe into the train and test sets.


from sklearn.model_selection import train_test_split


train_df,test_df = train_test_split(df,test_size = 0.7,random_state = 4)

# Create separate data-frames for the feature and target variables for both the train and test sets.


features = list(df.columns.values)                # creating list of feature variable 

features.remove('price')                          # removing target variale from the features


x_train = df[features]

x_test = df[features]

y_train = df['price']

y_test = df['price']

# Build a linear regression model using all the features to predict prices.


import statsmodels.api as sm


sm1 = sm.add_constant(x_train)


lin_reg = sm.OLS(y_train,sm1).fit()

lin_reg.params

# Print the summary of the linear regression report.

print(lin_reg.summary())

# Calculate N and p values

n = x_train.shape[0]

p = x_train.shape[1]

print('The value of N is : ',n)

print('The value of P is : ',p)

# Calculate the adjusted R-square value.

num = (1 - 0.685)*(n - 1)

den = (x_train.shape[0] - p -1 )

r_adj = 1-num/den

print('Value of adjusted R-squared value : ',r_adj)


# Build multiple linear regression model using all the features

from sklearn.linear_model import LinearRegression


lin_reg1 = LinearRegression()

lin_reg1.fit(x_train,y_train)


pred_train = lin_reg1.predict(x_train)


pred_test = lin_reg1.predict(x_test)


print(lin_reg1.coef_)

# Evaluating the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,mean_squared_log_error

print('Training set :')

print('r2_score :',r2_score(y_train,pred_train))

print('mean_squared_error :',mean_squared_error(y_train,pred_train))

print('Mean_absolute_error :',mean_absolute_error(y_train,pred_train))


print('\nTest set :')

print('r2_score :',r2_score(y_test,pred_test))

print('mean_squared_error :',mean_squared_error(y_test,pred_test))

print('Mean_absolute_error :',mean_absolute_error(y_test,pred_test))


# Creating a Python dictionary storing the moderately to highly correlated features with price and the corresponding correlation values.
# Keep correlation threshold to be 0.2

major_features = {}

for f in features:

  corr_coef = np.corrcoef(df['price'], df[f])[0, 1]

  if (corr_coef >= 0.2) or (corr_coef <= -0.2):

    major_features[f] = corr_coef

print("Number of features moderately to highly correlated with price =", len(major_features), "\n")

major_features

# Perform RFE and select best 7 features  

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

skl_lin_reg = LinearRegression()

rfe1 = RFE(skl_lin_reg, n_features_to_select = 7)

rfe1.fit(x_train[major_features.keys()], y_train)

print(major_features.keys(), "\n") # List of features out of which 7 best featuers are to be selected by RFE.

print(rfe1.support_, "\n") # Array containing the boolean values      

print(rfe1.ranking_) # Ranking of the features selected by RFE

# Print the 7 features selected by RFE in the previous step.

rfe_features = x_train[major_features.keys()].columns[rfe1.support_]

rfe_features

# Build multiple linear regression model using all the features selected after RFE

from sklearn.model_selection import train_test_split

x1 = df[rfe_features]

y1 = df['price']


# Split the DataFrame into the train and test sets such that test set has 33% of the values.

x1_train,x1_test,y1_train, y1_test = train_test_split(x1, y1, test_size = 0.33, random_state = 8)


# Build linear regression model using the 'sklearn.linear_model' module.

from sklearn.linear_model import LinearRegression


lin_reg2 = LinearRegression()

lin_reg2.fit(x1_train, y1_train)


# Print the value of the intercept

print('Intercept : ',lin_reg2.intercept_,'\n')


# Print the names of the features along with the values of their corresponding coefficients.

print('Features and their corresponding coefficients :\n')

for item in list(zip(x1.columns.values, lin_reg2.coef_)):

  print(f"{item[0]}".ljust(15, " "), f"{item[1]:.6f}")

# Evaluate the linear regression model using the 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


y1_train_pred = lin_reg2.predict(x1_train)

y1_test_pred = lin_reg2.predict(x1_test)


print(f"Train Set\n{'-' * 50}")

print(f"R-squared: {r2_score(y1_train, y1_train_pred):.3f}")

print(f"Mean Squared Error: {mean_squared_error(y1_train, y1_train_pred):.3f}")

print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y1_train, y1_train_pred)):.3f}")

print(f"Mean Absolute Error: {mean_absolute_error(y1_train, y1_train_pred):.3f}")
      

print(f"\n\nTest Set\n{'-' * 50}")


print(f"R-squared: {r2_score(y1_test, y1_test_pred):.3f}")

print(f"Mean Squared Error: {mean_squared_error(y1_test, y1_test_pred):.3f}")

print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y1_test, y1_test_pred)):.3f}")

print(f"Mean Absolute Error: {mean_absolute_error(y1_test, y1_test_pred):.3f}")



# Create a histogram for the errors obtained in the predicted values for the train set.

errors_train = y1_train - y1_train_pred


plt.figure(figsize = (19,8))

plt.hist(x = errors_train,bins = 'sturges',edgecolor = 'red')

plt.axvline(errors_train.mean(),color = 'red',label = 'Mean')

plt.title('Histogram on errrors in training set.')

plt.xlabel('Training set error.')

plt.legend()

plt.show()

#  a histogram for the errors obtained in the predicted values for the test set.

errors_test = y1_test - y1_test_pred


plt.figure(figsize = (19,8))

plt.hist(x = errors_test,bins = 'sturges',edgecolor = 'red')

plt.axvline(errors_test.mean(),color = 'red',label = 'Mean')

plt.title('Histogram on errrors in test set.')

plt.xlabel('Test set error.')

plt.legend()

plt.show()

# a scatter plot between the errors and the dependent variable for the train set.

plt.figure(figsize= (19,9))

plt.scatter(y1_train,errors_train)

plt.title('Scatter plot to check Homoscedasticity.')

plt.axhline(errors_train.mean(),color = 'red',label = 'Mean')

plt.legend()

plt.show()
