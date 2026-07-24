#!/usr/bin/env python
# coding: utf-8

# # Lab 2: ML Life Cycle: Data Understanding and Data Preparation

# In[9]:


import os
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns


# ## Business Brief
# 
# Read through the scenario below. You will be putting yourself in the shoes of a junior ML engineer (MLE) at NestIQ, a company tasked with developing a machine learning model to predict the prices of new short-term rentals for one of its clients.
# 
# ### Company and Context
# NestIQ is a data and analytics company that develops machine learning models to help real estate and short-term rental clients make better pricing decisions.
# 
# ### Business Challenge
# NestIQ's new client is a short-term rental operator based in New York City (NYC). They currently set prices manually or by using simple rules. For example, they start with a base price and increase it by a fixed amount for each additional bedroom, adjust for room type, and apply small premiums for highly-rated hosts and location. These rules provide a starting point, but pricing is often adjusted manually on a case-by-case basis. As the company grows, manual pricing becomes difficult to scale.
# 
# ### Business Goal
# NestIQ’s goal is to develop a pricing model for the client that estimates nightly listing prices for new short-term rentals. The supervised machine learning model should learn relationships between listing features and prices from the client’s current NYC short-term rental listings data and accurately predict the price of a new listing.
# 
# ### Your Role and Task
# You have just joined NestIQ as a junior MLE on the Pricing Model team. The team has been given a dataset of the operator’s current NYC short-term rental listings. It will use this data to train the pricing model.
# 
# Your task is to explore the data and prepare it for modeling. The choices you make during this process — such as how to handle missing values and outliers — will affect the information available to the model during training and ultimately its performance.
# 
# This work involves making decisions about the data, for example:
# 
# * Dropping columns with many missing values might remove useful information.
# * Modifying outliers may alter meaningful patterns in the data.
# 
# ### Technical Focus in This Lab
# 
# In this lab, you will practice the second and third steps of the machine learning life cycle: data understanding and data preparation. You will beging preparing your data so that it can be used to train a machine learning model that solves a regression problem. Note that by the end of the lab, your data set won't be completely ready for the modeling phase, but you will gain experience using some common data preparation techniques. 
# 
# You will use the Airbnb "listings" data set to represent the type of data NestIQ would receive from its short-term rental client.
# 
# You will complete the following tasks to transform your data:
# 
# 1. Build your data matrix and define your ML problem:
#     * Load the Airbnb "listings" data set into a DataFrame and inspect the data
#     * Define the label and convert the label's data type to one that is more suitable for modeling
#     * Identify features
# 2. Clean your data:
#     * Handle outliers by building a new regression label column by winsorizing outliers
#     * Handle missing data by replacing all missing values in the dataset with means
# 3. Perform feature transformation using one-hot encoding
# 4. Explore your data:
#     * Identify two features with the highest correlation with label
#     * Build appropriate bivariate plots to visualize the correlations between features and the label
# 5. Analysis:
#     * Analyze the relationship between the features and the label
#     * Brainstorm what else needs to be done to fully prepare the data for modeling

# ## Part 1. Build Your Data Matrix (DataFrame) and Define Your ML Problem

# #### Load a Data Set and Save it as a Pandas DataFrame

# We will be working with the Airbnb NYC "listings" data set. Use the specified path and name of the file to load the data. Save it as a Pandas DataFrame called `df`.

# In[10]:


# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "airbnbData.csv")


# **Task**: Load the data and save it to DataFrame `df`.
# 
# <i>Note:</i> You may receive a warning message. Ignore this warning.

# In[11]:


df = pd.read_csv(filename)


# ####  Inspect the Data
# 

# <b>Task</b>: Display the shape of `df` &mdash; that is, the number of rows and columns.

# In[12]:


df.shape


# <b>Task</b>: Display the column names.

# In[13]:


list(df.columns)


# **Task**: Get a peek at the data by displaying the first few rows, as you usually do.

# In[14]:


df.head()


# #### Define the Label

# Assume that your goal is to train a machine learning model that predicts the price of an Airbnb. This is an example of supervised learning and is a regression problem. In our dataset, our label will be the `price` column. Let's inspect the values in the `price` column.

# In[15]:


df['price']


# Notice the `price` column contains values that are listed as $<$currency_name$>$$<$numeric_value$>$. 
# <br>For example, it contains values that look like this: `$120`. <br>
# 
# **Task**:  Obtain the data type of the values in this column:

# In[16]:


df['price'].dtype


# Notice that the data type is "object," which in Pandas translates to the String data type.

# **Task**:  Display the first 15 unique values of  the `price` column:

# In[17]:


df['price'].head(15).unique()


# In order for us to use the prices for modeling, we will have to transform the values in the `price` column from strings to floats. We will:
# 
# * remove the dollar signs (in this case, the platform forces the currency to be the USD, so we do not need to worry about targeting, say, the Japanese Yen sign, nor about converting the values into USD). 
# * remove the commas from all values that are in the thousands or above: for example, `$2,500`. 
# 
# The code cell below accomplishes this.

# In[18]:


df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].astype(float)


# **Task**:  Display the first 15 unique values of  the `price` column again to make sure they have been transformed.

# In[19]:


df['price'].head(15).unique()


# #### Identify Features

# Simply by inspecting the data, let's identify some columns that should not serve as features &mdash; those that will not help us solve our predictive ML problem. 

# Some that stand out are columns that contain website addresses (URLs).
# 
# **Task**: Create a list which contains the names of columns that contain URLs. Save the resulting list to variable `url_colnames`.
# 
# *Tip*: There are different ways to accomplish this, including using Python list comprehensions.

# In[20]:


url_colnames = []
for columns in df.columns:
    if('url' in columns):
        url_colnames.append(columns)
url_colnames


# **Task**: Drop the columns with the specified names contained in list `url_colnames` in place (that is, make sure this change applies to the original DataFrame `df`, instead of creating a temporary new DataFrame object with fewer columns).

# In[21]:


df = df.drop(columns=url_colnames)


# **Task**: Display the shape of the data to verify that the new number of columns is what you expected.

# In[22]:


df.shape


# **Task**: In the code cell below, display the features that we will use to solve our ML problem.

# In[23]:


df.head()


# **Task**: Are there any other features that you think may not be well suited for our machine learning problem? Note your findings in the markdown cell below.

# The features that would not suit well with out machine learning problem are those with low correlation with other features. Some features that does not suit well are the description column and the last scraped columns because the dates are all the same for each house and house description wouldn't serve much in helping to predict the price of a short-term rental. 

# ## Part 2. Clean Your Data
# 
# Let's now handle outliers and missing data.

# ### a. Handle Outliers
# 
# Let us prepare the data in our label column. Namely, we will detect and replace outliers in the data using winsorization.

# **Task**: Create a new version of the `price` column, named `label_price`, in which you will replace the top and bottom 1% outlier values with the corresponding percentile value. Add this new column to the DataFrame `df`.

# Remember, you will first need to load the `stats` module from the `scipy` package:

# In[24]:


import scipy.stats as stats
df['label_price'] = stats.mstats.winsorize(df['price'], limits=[0.01,0.01])


# Let's verify that the new column `label_price` was added to DataFrame `df`:

# In[25]:


df.head()


# **Task**: Check that the values of `price` and `label_price` are *not* identical. 
# 
# You will do this by subtracting the two columns and finding the resulting *unique values*  of the resulting difference. <br>Note: If all values are identical, the difference would not contain unique values. If this is the case, outlier removal did not work.

# In[26]:


df['difference'] = np.sum(df['price']) - np.sum(df['label_price'])
list(df['difference'].unique())


# ### b. Handle Missing Data
# 
# Next we are going to find missing values in our entire dataset and impute the missing values by
# replace them with means.

# #### Identifying missingness

# **Task**: Check if a given value in the data is missing, and sum up the resulting values by columns. Save this sum to variable `nan_count`. Print the results.

# In[27]:


nan_count = np.sum(df.isnull(), axis = 0)
nan_count


# Those are more columns than we can eyeball! For this exercise, we don't care about the number of missing values -- we just want to get a list of columns that have *any* missing values.
# 
# <b>Task</b>: From the variable `nan_count`, create a new series called `nan_detected` that contains `True` or `False` values that indicate whether the number of missing values is *not zero*:

# In[28]:


nan_detected = nan_count != 0
nan_detected


# Since replacing the missing values with the mean only makes sense for the columns that contain numerical values (and not for strings), let us create another condition: the *type* of the column must be `int` or `float`.

# **Task**: Create a series that contains `True` if the type of the column is either `int64` or `float64`. Save the results to the variable `is_int_or_float`.

# In[29]:


is_int_or_float = (df.dtypes == 'int64') | (df.dtypes == 'float64')
is_int_or_float


# <b>Task</b>: Combine the two binary series (`nan_detected` and `is_int_or_float`) into a new series named `to_impute`. It will contain the value `True` if a column contains missing values *and* is of type 'int' or 'float'

# In[30]:


to_impute = nan_detected & is_int_or_float
to_impute


# Finally, let's display a list that contains just the selected column names contained in `to_impute`:

# In[31]:


df.columns[to_impute]


# We just identified and displayed the list of candidate columns for potentially replacing missing values with the column mean.

# Assume that you have decided that you should impute the values for these specific columns: `host_listings_count`, `host_total_listings_count`, `bathrooms`, `bedrooms`, and `beds`:

# In[32]:


to_impute_selected = ['host_listings_count', 'host_total_listings_count', 'bathrooms',
       'bedrooms', 'beds']


# #### Keeping record of the missingness: creating dummy variables 

# As a first step, you will now create dummy variables indicating the missingness of the values.

# **Task**: For every column listed in `to_impute_selected`, create a new corresponding column called `<original-column-name>_na`. These columns should contain the a `True`or `False` value in place of `NaN`.

# In[33]:


for columns in to_impute_selected:
    df[columns + "_na"] = df[columns].isnull()


# Check that the DataFrame contains the new variables:

# In[34]:


df.head()


# #### Replacing the missing values with mean values of the column

# **Task**: For every column listed in `to_impute_selected`, fill the missing values with the corresponding mean of all values in the column (do not create new columns).

# In[35]:


for columns in to_impute_selected:
    mean_val = df[columns].mean()
    df[columns].fillna(value=mean_val, inplace=True)


# Check your results below. The code displays the count of missing values for each of the selected columns. 

# In[36]:


for colname in to_impute_selected:
    print("{} missing values count :{}".format(colname, np.sum(df[colname].isnull(), axis = 0)))


# Why did the `bathrooms` column retain missing values after our imputation?

# **Task**: List the unique values of the `bathrooms` column.

# In[37]:


df['bathrooms'].unique()


# The column did not contain a single value (except the `NaN` indicator) to begin with.

# ## Part 3. Perform One-Hot Encoding

# Machine learning algorithms operate on numerical inputs. Therefore, we have to transform text data into some form of numerical representation to prepare our data for the model training phase. Some features that contain text data are categorical. Others are not. For example, we removed all of the features that contained URLs. These features were not categorical, but rather contained what is called unstructured text. However, not all features that contain unstructured text should be removed, as they can contain useful information for our machine learning problem. Unstructured text data is usually handled by Natural Language Processing (NLP) techniques. You will learn more about NLP later in this course. 
# 
# However, for features that contain categorical values, one-hot encoding is a common feature engineering technique that transforms them into binary representations. 

# We will first choose one feature column to one-hot encode: `host_response_time`. Let's inspect the unique values this feature can have. 

# In[38]:


df['host_response_time'].unique()


# Note that each entry can contain one of five possible values. 
# 
# **Task**: Since one of these values is `NaN`, replace every entry in the column `host_response_time` that contains a `NaN` value with the string 'unavailable'.

# In[39]:


df['host_response_time'].fillna(value='unavailable', inplace=True)


# Let's inspect the `host_response_time` column to see the new values.

# In[40]:


df['host_response_time'].unique()


# **Task**: Use `pd.get_dummies()` to one-hot encode the `host_response_time` column. Save the result to DataFrame `df_host_response_time`. 

# In[41]:


df_host_response_time = pd.get_dummies(df['host_response_time'])
df_host_response_time


# **Task**: Since the `pd.get_dummies()` function returned a new DataFrame rather than making the changes to the original DataFrame `df`, add the new DataFrame `df_host_response_time` to DataFrame `df`, and delete the original `host_response_time` column from DataFrame `df`.
# 

# In[42]:


df = pd.concat([df,df_host_response_time])
df.drop(columns='host_response_time', inplace=True)


# Let's inspect DataFrame `df` to see the changes that have been made.

# In[43]:


df.columns


# #### One-hot encode additional features
# 
# **Task**: Use the code cell below to find columns that contain string values  (the 'object' data type) and inspect the *number* of unique values each column has.

# In[44]:


for columns in df.columns:
    if df[columns].dtype == 'object':
        print(columns)
        print(df[columns].nunique())


# **Task**: Based on your findings, identify features that you think should be transformed using one-hot encoding.
# 
# 1. Use the code cell below to inspect the unique *values* that each of these features have.

# In[45]:


for columns in df.columns:
    if df[columns].dtype == 'object':
        print(df[columns].unique())


# 2.  List these features and explain why they would be suitable for one-hot encoding. Note your findings in the markdown cell below.

# The features that would be suitable for one-hot encoding are the columns that had low number of unique values such as host_listings_count_na, host_total_listings_count_na, bathrooms_na, bedrooms_na, beds_na, room_type, and host_identity_verified.They are suitable because since their unique values are low, their categorical values could be simplified as binary classification(0s and 1s).
# 

# **Task**: In the code cell below, one-hot encode one of the features you have identified and replace the original column in DataFrame `df` with the new one-hot encoded columns. 

# In[46]:


df['room_type'].unique()
df['room_type'].fillna(value='unavailable', inplace=True)
df_room_type = pd.get_dummies(df['room_type'])
df = pd.concat([df,df_room_type])
df.drop(columns='room_type', inplace=True)


# ## Part 4. Explore Your Data

# You will now perform exploratory data analysis in preparation for selecting your features as part of feature engineering. 
# 
# #### Identify Correlations
# 
# We will focus on identifying which features in the data have the highest correlation with the label.

# Let's first run the `corr()` method on DataFrame `df` and save the result to the variable `corr_matrix`. Let's round the resulting correlations to five decimal places:

# In[47]:


corr_matrix = round(df.corr(),5)
corr_matrix


# The result is a computed *correlation matrix*. The values on the diagonal are all equal to 1 because they represent the correlations between each column with itself. The matrix is symmetrical with respect to the diagonal.<br>
# 
# We only need to observe correlations of all features with the column `label_price` (as opposed to every possible pairwise correlation). Se let's query the `label_price` column of this matrix:
# 
# **Task**: Extract the `label_price` column of the correlation matrix and save the results to the variable `corrs`.

# In[48]:


corrs = corr_matrix['label_price']
corrs


# **Task**: Sort the values of the series we just obtained in the descending order and save the results to the variable `corrs_sorted`.

# In[49]:


corrs_sorted = corrs.sort_values(ascending=False)
corrs_sorted


# **Task**: Use Pandas indexing to extract the column names for the top two correlation values and save the results to the Python list `top_two_corr`. Add the feature names to the list in the order in which they appear in the output above. <br> 
# 
# <b>Note</b>: Do not count the correlation of `label` column with itself, nor the `price` column -- which is the `label` column prior to outlier removal.

# In[50]:


top_two_corr = list(corrs_sorted.iloc[[2,3]].index)
top_two_corr


# #### Bivariate Plotting: Produce Plots for the Label and Its Top Correlates
# 
# Let us visualize our data.

# We will use the `pairplot()` function in `seaborn` to plot the relationships between the two features and the label.

# **Task**: Create a DataFrame `df_corrs` that contains only three columns from DataFrame `df`: the label, and the two columns which correlate with it the most.

# In[51]:


df_corrs = pd.DataFrame({'label_price': df['label_price'], 'accommodates': df['accommodates'], 'bedrooms': df['bedrooms']})
df_corrs


# **Task**: Create a `seaborn` pairplot of the data subset you just created. Specify the *kernel density estimator* as the kind of the plot, and make sure that you don't plot redundant plots.
# 
# <i>Note</i>: It will take a few minutes to run and produce a plot.

# In[ ]:


sns.pairplot(data=df_corrs, kind = 'kde', corner=True)


# ## Part 5. Analysis
# 
# 1. In Part 3, you winsorized the price column and imputed missing values in bedrooms, bathrooms, and beds using column means. What are the risks of these techniques, and how could those risks affect NestIQ's price estimates?
# 2. In Part 5, you identified the two features most correlated with `label_price` and visualized them against the label in a pairplot. Are these features strongly or weakly correlated with the label? Are they features that should be used for our predictive machine learning problem? Additionally, Name one factor that probably influences a listing price in the real world but is not captured in this dataset. How might that gap affect NestIQ's model?
# 3. Inspect your data matrix. It has a few features that contain unstructured text, meaning text data that is neither numerical nor categorical. List some features that contain unstructured text that you think are valuable for our predictive machine learning problem. Are there other remaining features that you think need to be prepared for the modeling phase? Do you have any suggestions on how to prepare these features?
# 
# Record your findings in the cell below.

# 1. The risks to these techniques are that the replacing the missing values with the column's mean. Those risks affect NestIQ's price estimates by potentially messing up the prediction model, trained with those mean values but not accurately predicting the next price of a short-term housing.
# 2. The two features are weakly correlated with the label because the I don't believe the two features would serve well in calculating the price of the next housing. Therefore, the features should not be used for our predictive ML problem. One factor that could prove useful is the a feature that lists the rental period. Knowing the period of housing rental would be helpful in predicting the price because the longer the rental period, the more money it could cost for the price. The gap might affect the NestIQ's model by not considering the rental period and that could influence in how expensive a housing might be. 
# 3. Some features that are valuable for our ML problem would be the name and description columns of type of housing. They could be useful in our ML problem by potentially seeking out more datasets of various types of housing in NYC. As of right now, I can't think of anything that needs to be prepared for the modeling phase other than the rental period column. To prepare these features are to follow the steps from this lab from one-hot-encoding, replacing outliers, and inspecting our data matrix to remove columns that won't serve well in our ML problem.

# ## Part 6. Reflection: AI Usage
# 
# 1. Did you use AI tools for this lab? If yes, which ones and at what points in your work? If no, briefly explain your reasoning.
# 2. If you used AI, describe one specific prompt that was useful and explain why it worked. If you did not use AI, walk through one part of the lab where you had to figure something out on your own and explain how you got there.
# 3. How did you verify that your work was correct? What would you look for to catch a mistake, whether it came from AI or from your own reasoning?
# 4. What is one thing you would do differently next time, either in how you approached the lab or in how you used (or did not use) AI?
# 
# 
# Record your findings in the cell below.

# 1. Yes, I used Google Gemini and other documentation tools to mainly remind me of how certain pandas functions work such as .replace(), .corr(), fillna(), and sort_values(). 
# 2. In a part of this lab I was computing the to_impute variable and I used the && operator to compare the two variables is_float and is_int64. However, I got a long error when I tried to compile the cell. Therefore, I asked Gemini why am I getting this error. Gemini suggested that Python AND operator is & but not &&. It worked because once i run the cell again, the expected output worked. I code in C++ and the AND operator is && but in Python, it's a single &. 
# 3. I verfied that my work was correct by reading the instructions again and compared my output and the expected output to see if they match. I would look for syntax errors then hit up Google to research more documentation and explanation of python libraries. 
# 4. One thing I would do differently next time in how I approached the lab is making sure I have examples from the unit 2 coding exercises in hand with me. Several instructions in this lab were similar to the problems from the Unit 2 module. Even though they are similar, I should also read more documentation of these methods online from Geeks for Geeks or W3Schools. 
