
# coding: utf-8

# # SF-DAT-21 | Unit Project 3
# 
# In this project, you will perform a logistic regression on the admissions data we've been working with in Unit Projects 1 and 2.

# In[179]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as pl
get_ipython().magic(u'matplotlib inline')


# In[180]:

df_raw = pd.read_csv("../../dataset/admissions.csv")
df = df_raw.dropna()
print df.head()


# ## Part 1. Frequency Tables
# 
# #### Question 1. Let's create a frequency table of our variables.

# In[181]:

# frequency table for prestige and whether or not someone was admitted
# Make a crosstab
# Name the count column


# In[182]:

print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])


# In[ ]:




# In[ ]:




# ## Part 2. Return of dummy variables

# #### Question 2.1. Create class or dummy variables for prestige.

# In[183]:

ys = pd.get_dummies(df_raw.prestige, prefix = None)


# In[184]:

ys


# #### Question 2.2. When modeling our class variables, how many do we need?

# Answer:

# ## Part 3. Hand calculating odds ratios
# 
# Develop your intuition about expected outcomes by hand calculating odds ratios.

# In[185]:

cols_to_keep = ['admit', 'gre', 'gpa']
handCalc = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_1':])
print handCalc.head()


# In[186]:

# crosstab prestige 1 admission
# frequency table cutting prestige and whether or not someone was admitted


# In[187]:

dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
print dummy_ranks.head()


# In[188]:

print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])


# In[ ]:




# In[ ]:




# #### Question 3.1. Use the cross tab above to calculate the odds of being admitted to grad school if you attended a #1 ranked college.

# In[ ]:




# #### Question 3.2. Now calculate the odds of admission if you did not attend a #1 ranked college.

# In[ ]:




# #### Question 3.3. Calculate the odds ratio.

# In[ ]:




# #### Question 3.4. Write this finding in a sentenance:

# Answer:

# #### Question 3.5. Print the cross tab for prestige_4.

# In[190]:

print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])


# #### Question 3.6. Calculate the OR.

# In[ ]:




# #### Question 3.7. Write this finding in a sentence.

# Answer:

# ## Part 4. Analysis

# In[160]:

# create a clean data frame for the regression
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
print data.head()


# We're going to add a constant term for our Logistic Regression.  The statsmodels function we're going to be using requires that intercepts/constants are specified explicitly.

# In[161]:

# manually add the intercept
data['intercept'] = 1.0


# #### Question 4.1. Set the covariates to a variable called train_cols.

# In[162]:

train_cols = data.columns[1:]
# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

logit = sm.Logit(data['admit'], data[train_cols])


# #### Question 4.2. Fit the model.

# In[163]:

# fit the model
result = logit.fit()


# #### Question 4.3. Print the summary results.

# In[164]:

print result.summary()


# #### Question 4.4. Calculate the odds ratios of the coeffincients and their 95% CI intervals
# 
# hint 1: np.exp(X)
# 
# hint 2: conf['OR'] = params
# 
#         conf.columns = ['2.5%', '97.5%', 'OR']

# In[165]:

print np.exp(result.params)


# In[166]:

params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)


# #### Question 4.5. Interpret the OR of Prestige_2.

# Answer:0.506548

# #### Question 4.6. Interpret the OR of GPA.

# Answer:2.180027

# ## Part 5: Predicted probablities
# 

# As a way of evaluating our classifier, we're going to recreate the dataset with every logical combination of input values.  This will allow us to see how the predicted probability of admission increases/decreases across different variables.  First we're going to generate the combinations using a helper function called cartesian (above).
# 
# We're going to use np.linspace to create a range of values for "gre" and "gpa".  This creates a range of linearly spaced values from a specified min and maximum value--in our case just the min/max observed values.

# In[167]:

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


# In[168]:

# instead of generating all possible values of GRE and GPA, we're going
# to use an evenly spaced range of 10 values from the min to the max
gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)

print gres
# array([ 220.        ,  284.44444444,  348.88888889,  413.33333333,
#         477.77777778,  542.22222222,  606.66666667,  671.11111111,
#         735.55555556,  800.        ])

gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)

print gpas
# array([ 2.26      ,  2.45333333,  2.64666667,  2.84      ,  3.03333333,
#         3.22666667,  3.42      ,  3.61333333,  3.80666667,  4.        ])

# enumerate all possibilities
combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))


# #### Question 5.1. Recreate the dummy variables.

# In[169]:

# recreate the dummy variables

# keep only what we need for making predictions


# In[173]:

combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))
# recreate the dummy variables
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
dummy_ranks = pd.get_dummies(combos['prestige'], prefix='prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']

# keep only what we need for making predictions
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])


# #### Question 5.2. Make predictions on the enumerated dataset.

# In[174]:

combos['admit_pred'] = result.predict(combos[train_cols])


# #### Question 5.3. Interpret findings for the last 4 observations.

# Answer:

# ## Bonus
# 
# Plot the probability of being admitted into graduate school, stratified by GPA and GRE score.

# In[ ]:



