#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[3]:


from sklearn.preprocessing import StandardScaler


# In[4]:


df=pd.read_csv('watson_healthcare_modified.csv')


# In[5]:


import plotly.express as px


# In[6]:


#pip install plotly


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


#handling missing values
 # Checking for total null values
df.isnull().sum()


# In[10]:


# Get duplicates
df[df.duplicated()] 


# In[11]:


# Remove Duplicates
df.drop_duplicates(inplace=True)


# In[12]:


#Outlier detection and removal
num_col = df.select_dtypes(include=np.number)
cat_col = df.select_dtypes(exclude=np.number)


# In[13]:


plt.figure(figsize=(30,20))
for index,column in enumerate(num_col):
    plt.subplot(7,4,index+1)
    sns.boxplot(data=num_col,x=column)
    
plt.tight_layout(pad = 1.0)


# In[14]:


num_col = num_col.drop("EmployeeID",axis = 1)


# In[15]:


num_col = num_col.drop(["DailyRate","MonthlyIncome","MonthlyRate"],axis = 1)


# In[16]:


df.info()


# In[17]:


#eda
plt.figure(figsize=(30,50))
for index,column in enumerate(num_col):
    plt.subplot(5,5,index+1)
    sns.countplot(data=num_col,x=column)
    plt.xticks(rotation = 90)
plt.tight_layout(pad = 1.0)
plt.show()


# In[18]:


#attr dominates all
num_col["Attrition"] = df.Attrition
fig = plt.figure(figsize=(20,20))
for index in range(len(num_col.columns)):
    plt.subplot(6,5,index + 1)
    sns.scatterplot(x = num_col.iloc[:,index],y="Attrition",data = num_col)
    
    
fig.tight_layout(pad = 1.0)

#o/p no regression observed


# In[19]:


plt.figure(figsize=(30,50))
for index,column in enumerate(num_col):
    plt.subplot(5,5,index+1)
    sns.histplot(data=num_col,x=column,kde=True)
    plt.xticks(rotation = 90)
plt.tight_layout(pad = 1.0)
plt.show()


# In[20]:


#Observations : EmployeeCount and StandartHours has one value.We should drop them because variables which have one value will not give us good information.
df.drop(["EmployeeID","EmployeeCount","StandardHours"],axis = 1,inplace = True)


# In[21]:


plt.figure(figsize=(25,20))
for index,column in enumerate(cat_col):
    plt.subplot(3,3,index+1)
    sns.countplot(data=cat_col,x=column)
    
plt.tight_layout(pad = 1.0)
plt.show()


# In[22]:


px.histogram(df,x="Department",color="Attrition",barmode="group",text_auto=".2f",template="plotly_dark",
             title = "Percentage of Department Type")


# In[23]:


plt.figure(figsize=(25,20))
for index,column in enumerate(cat_col):
    plt.subplot(3,3,index+1)
    sns.countplot(data=cat_col,x=column,hue = cat_col.Attrition,palette="magma")
    
plt.tight_layout(pad = 1.0)
plt.show()


# In[24]:


df.drop("Over18",axis = 1,inplace = True)


# In[25]:


genderAttrition = df[["Attrition","Gender"]]


# In[26]:


NoAttrition = genderAttrition.query('Attrition == "No"')
YesAttrirition = genderAttrition.query('Attrition == "Yes"')
YesAttrirition.Attrition.replace("Yes",1,inplace=True)
YesAttrirition.Gender.replace("Male",0,inplace=True)
YesAttrirition.Gender.replace("Female",1,inplace=True)
NoAttrition.Gender.replace("Male",0,inplace=True)
NoAttrition.Gender.replace("Female",1,inplace=True)
YesAttrirition


# In[27]:


fig,axes = plt.subplots(1,2,figsize = (15,8),sharey=True)

YesAttrirition.Gender.value_counts().plot(ax = axes[0],kind = "pie", autopct = "%.0f%%")
axes[0].set_title("Being Attrition on Gender (0:Male 1:Female)")

NoAttrition.Gender.value_counts().plot(ax = axes[1],kind = "pie", autopct = "%.0f%%")
axes[1].set_title("Being Attrition on Gender (0:Male 1:Female)")


# In[28]:


overTimeAttrition = df[["Attrition","OverTime"]]
NoAttrition = overTimeAttrition.query('Attrition == "No"')
YesAttrirition = overTimeAttrition.query('Attrition == "Yes"')

YesAttrirition.OverTime.replace("No",0,inplace=True)
YesAttrirition.OverTime.replace("Yes",1,inplace=True)

NoAttrition.OverTime.replace("No",0,inplace=True)
NoAttrition.OverTime.replace("Yes",1,inplace=True)


# In[29]:


fig,axes = plt.subplots(1,2,figsize = (15,8),sharey=True)

YesAttrirition.OverTime.value_counts().plot(ax = axes[0],kind = "pie", autopct = "%.0f%%")
axes[0].set_title("Being Attrition on Overtime (0:No 1:Yes)")

NoAttrition.OverTime.value_counts().plot(ax = axes[1],kind = "pie", autopct = "%.0f%%")
axes[1].set_title("Being Attrition on Gender (0:No 1:Yes)")


# In[30]:


correlation = df.corr()


# In[31]:


plt.figure(figsize = (15,8))
sns.heatmap(correlation,mask = correlation < 0.7,annot=True)


# In[32]:


#Observations :There are a lot of correlation.We can drop them for faster calculating.
#Is there any problem to droping correlation values? No,because value will carry any other one which is correlation each
#other feature.


# In[33]:


df = df.drop(["MonthlyIncome","TotalWorkingYears","PercentSalaryHike","YearsInCurrentRole","YearsWithCurrManager"],axis=1)


# In[34]:


#cols reduced to only 26
#preprocesssing
X = df.drop("Attrition",axis = 1)
y =  df["Attrition"]


# In[35]:


cat_value = X.select_dtypes(exclude=np.number)


# In[36]:


X = pd.get_dummies(X,columns=cat_value.columns)


# In[37]:


y.replace("No",0,inplace=True)
y.replace("Yes",1,inplace=True)


# In[38]:


scaler = StandardScaler()


# In[39]:


xColumns = X.columns
X = scaler.fit_transform(X)
X = pd.DataFrame(X,columns=xColumns)


# In[40]:


X
#Observations : We make a standart scaler because our values are close to each other in that case we should use standart scaler.
#Otherwise,when values are distant each other we use min max.


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[43]:


rd = RandomForestClassifier()


# In[44]:


rd.fit(X_train,y_train)


# In[45]:


y_pred = rd.predict(X_test)


# In[46]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[47]:


parameters = {
    "n_estimators" : [50,100,200,400],
    "criterion" : ["gini","entropy","log_loss"],
    "max_depth" : [1,5,10,None],
    "max_features" : ["sqrt","log2",None]
    
}


# In[48]:


from sklearn.model_selection import GridSearchCV
rsc = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=parameters,
                         cv = 5,
                         verbose=1,
                         scoring="accuracy"
                        )


# In[49]:


rsc.fit(X_train,y_train)


# In[50]:


print("Best parameters : ",rsc.best_params_)
print("Best scor {:.2f} ".format(rsc.best_score_))


# In[ ]:





# In[ ]:




