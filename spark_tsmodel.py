#!/usr/bin/env python
##########libraries###########
import os
import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf
from pyspark.sql.functions import percent_rank
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler,VectorIndexer
from pyspark.sql.functions import broadcast
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn import svm
import datetime
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import mean_squared_error
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split

##########sparkcontext###############

sc= SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

print("started")

###########RMSLE-calusulation########

def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

##########Reading of imput Data###########

customer = sqlContext.read.csv('hdfs://192.168.11.70:/user/root/customer.csv', header=True, inferSchema = True)
customer =customer.select("*").toPandas()
lineitem = sqlContext.read.csv('hdfs://192.168.11.70:/user/root/lineitem.csv', header=True, inferSchema = True)
lineitem =lineitem.select("*").toPandas()
order = sqlContext.read.csv('hdfs://192.168.11.70:/user/root/orders.csv', header=True, inferSchema = True)
order =order.select("*").toPandas()

print("data has been read")

###########ETL############################

sales = order.join(customer, order.o_custkey == customer.c_custkey, how = 'inner')
sales = sales.sort_index()

sales.columns = ['key_old', 'o_orderdate', 'o_orderkey', 'o_custkey', 'o_orderpriority',
       'o_shippriority', 'o_clerk', 'o_orderstatus', 'o_totalprice',
       'o_comment', 'c_custkey', 'c_mktsegment', 'c_nationkey', 'c_name',
       'c_address', 'c_phone', 'c_acctbal', 'c_comment']

sales2 = sales.join(lineitem,sales.o_orderkey == lineitem.l_orderkey, how = 'outer')

sales3 = sales2.groupby(by = 'o_orderdate')

sales4 = sales3.agg({'l_quantity': 'sum'})# .withColumnRenamed("sum(l_quantity)", "TOTAL_SALES") .withColumnRenamed("o_orderdate", "ORDERDATE")

print("End of ETL pipeline")


orderdates = pd.to_datetime(sales4.index.values)

orderdates = [datetime.datetime(i.year, i.month, i.day,) for i in orderdates]

l = []
l2 = []


for i in orderdates:
    l = []
    l.append(i.timestamp())
    l.append(i.day)
    l.append(i.timetuple().tm_wday)
    l.append(i.timetuple().tm_yday)
    l.append(i.isocalendar()[1])
    l2.append(l)

print("dateconverted")

tmp = np.array(sales4.values)
tmp = tmp.reshape(tmp.shape[0],)
data_new = pd.DataFrame()
data_new['SALES'] = tmp
data_new[['DATE','DAY','WDAY','YDAY','WEEK']] = pd.DataFrame(np.array(l2))
data_new['ONES'] = np.ones((len(data_new)))

print("converted to datframe")

X = np.array(data_new[['DATE','DAY','WDAY','YDAY','WEEK','ONES']])
X = X.reshape(X.shape[0],X.shape[1])
Y = np.array(data_new[['SALES']])
Y = Y.reshape(Y.shape[0],1)

cutoff = 0.1
length = int((1-cutoff)*(len(X)))
X_train = X[0:length]
X_test = X[length:len(X)]

Y_train = Y[0:length]
Y_test = Y[length:len(Y)]

print("pre-processingdone")

weights  = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train)),X_train.T),Y_train)

print("model Ready")

Y_pred = np.dot(X_test,weights)


Y_pred = Y_pred.reshape(Y_pred.shape[0],)
Y_test = Y_test.reshape(Y_test.shape[0],)

print("predictions done")

RMSE = np.sqrt(np.mean((Y_test-Y_pred)**2))
RMSLE = rmsle(Y_test,Y_pred)
print(RMSE)
print(RMSLE)

sc.stop()
