import os
import sys
import time
import datetime

import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np

############################## tools #############################################
# check how many values are null in one dataframe
def null_hunter(df):
    df_cols = df.columns.values
    print(df_cols)
    for c in df_cols:
        NaN_lenth = len([i for i in df[c] if pd.isnull(i)])
        if NaN_lenth!=0:
            print(c,'miss',NaN_lenth)


def min_store_num(df):
    lenth_store = [df[df.Store==i] for i in range(1,1116)]
    return min(lenth_store)


############################## features ready for ml ##############################
simple_feature = ['Store', 'Open', 'Promo',
                  'SchoolHoliday', 'Customers', 'Sales']

def ssf(good_df, store_raw,
        simple_store_feature=['CompetitionDistance', 'Promo2']):
    store_list = good_df.Store.values
    for feature in simple_store_feature:
        store_feature_list = store_raw[feature].values
        good_df_feature = [store_feature_list[i - 1] for i in store_list]
        good_df[feature] = good_df_feature
    return good_df


############################## features need OHE ##############################
## stateholiday
def OHE_stateholiday(df_raw, good_df):
    stateholiday = df_raw.StateHoliday.values
    is_state_holiday = []
    is_public_holiday = []
    is_easter = []
    is_christmas = []
    for value in stateholiday:
        if value == '0' or value == 0:
            is_state_holiday.append(0)
            is_public_holiday.append(0)
            is_easter.append(0)
            is_christmas.append(0)
        elif value == 'a':
            is_state_holiday.append(1)
            is_public_holiday.append(1)
            is_easter.append(0)
            is_christmas.append(0)
        elif value == 'b':
            is_state_holiday.append(1)
            is_public_holiday.append(0)
            is_easter.append(1)
            is_christmas.append(0)
        elif value == 'c':
            is_state_holiday.append(1)
            is_public_holiday.append(0)
            is_easter.append(0)
            is_christmas.append(1)
        else:
            print(value)
            is_state_holiday.append(value)
            is_public_holiday.append(value)
            is_easter.append(value)
            is_christmas.append(value)

    good_df['is_state_holiday'] = is_state_holiday
    good_df['is_public_holiday'] = is_public_holiday
    good_df['is_easter'] = is_easter
    good_df['is_christmas'] = is_christmas

    return good_df


## storetype
def OHE_storetype(good_df, store_raw):
    store_list = good_df.Store.values
    storetype = store_raw.StoreType.values

    is_type_a = []
    is_type_b = []
    is_type_c = []
    is_type_d = []

    for store_idx in store_list:
        value = storetype[store_idx-1]
        if value=='a':
            is_type_a.append(1)
            is_type_b.append(0)
            is_type_c.append(0)
            is_type_d.append(0)
        elif value=='b':
            is_type_a.append(0)
            is_type_b.append(1)
            is_type_c.append(0)
            is_type_d.append(0)
        elif value=='c':
            is_type_a.append(0)
            is_type_b.append(0)
            is_type_c.append(1)
            is_type_d.append(0)
        elif value=='d':
            is_type_a.append(0)
            is_type_b.append(0)
            is_type_c.append(0)
            is_type_d.append(1)
        else:
            is_type_a.append(value)
            is_type_b.append(value)
            is_type_c.append(value)
            is_type_d.append(value)
            
    good_df['is_type_a'] = is_type_a
    good_df['is_type_b'] = is_type_b
    good_df['is_type_c'] = is_type_c
    good_df['is_type_d'] = is_type_d

    return good_df

## Assortment
def OHE_assortment(good_df, store_raw):
    store_list = good_df.Store.values
    storescale = store_raw.Assortment.values
    is_scale_a = []
    is_scale_b = []
    is_scale_c = []

    for store_idx in store_list:
        value = storescale[store_idx-1]
        if value=='a':
            is_scale_a.append(1)
            is_scale_b.append(0)
            is_scale_c.append(0)
        elif value=='b':
            is_scale_a.append(0)
            is_scale_b.append(1)
            is_scale_c.append(0)
        elif value=='c':
            is_scale_a.append(0)
            is_scale_b.append(0)
            is_scale_c.append(1)

        else:
            is_scale_a.append(value)
            is_scale_b.append(value)
            is_scale_c.append(value)
            
    good_df['is_scale_a'] = is_scale_a
    good_df['is_scale_b'] = is_scale_b
    good_df['is_scale_c'] = is_scale_c

    return good_df

## promointerval
def OHE_promointerval(good_df,store_raw):
    store_list = good_df.Store.values
    promointerval = store_raw.PromoInterval.values
    is_promo2_a = []
    is_promo2_b = []
    is_promo2_c = []

    for store_idx in store_list:
        value = promointerval[store_idx-1]
        if value=='Jan,Apr,Jul,Oct':
            is_promo2_a.append(1)
            is_promo2_b.append(0)
            is_promo2_c.append(0)
        elif value=='Feb,May,Aug,Nov':
            is_promo2_a.append(0)
            is_promo2_b.append(1)
            is_promo2_c.append(0)
        elif value=='Mar,Jun,Sept,Dec':
            is_promo2_a.append(0)
            is_promo2_b.append(0)
            is_promo2_c.append(1)
        else:
            is_promo2_a.append(value)
            is_promo2_b.append(value)
            is_promo2_c.append(value)
            
    good_df['is_promo2_a'] = is_promo2_a
    good_df['is_promo2_b'] = is_promo2_b
    good_df['is_promo2_c'] = is_promo2_c

    return good_df

############################## features need more work ##############################
## datetime
def good_datetime(df_raw,good_df):
    def resolve_date(date_str):
        date = date_str.split('-')
        month = int(date[1])
        day = int(date[2])
        year,woy,dow = datetime.date(int(date[0]),int(date[1]),int(date[2])).isocalendar()
        return year,month,day,woy,dow

    date_list = df_raw.Date.values
    year_list = []
    month_list = []
    day_list = []
    woy_list = []

    for i in date_list:
        if len(i)!=0:
            year,month,day,woy,dow = resolve_date(str(i))
            year_list.append(year)
            month_list.append(month)
            day_list.append(day)
            woy_list.append(woy)
        else:
            print(i)
            year_list.append(i)
            month_list.append(i)
            day_list.append(i)
            woy_list.append(i)

    good_df['Year']=year_list
    good_df['Month']=month_list
    good_df['DayOfMonth']=day_list
    good_df['WeekOfYear']=woy_list
    good_df['DayOfWeek']=df_raw.DayOfWeek

    return good_df

####### it is complicate
def competitiontime_promo2tim(good_df,store_raw):
    store_list = good_df.Store.values

    cs_year_list = store_raw.CompetitionOpenSinceYear.values
    cs_month_list = store_raw.CompetitionOpenSinceMonth.values

    cs_year_list_for_good_df = [cs_year_list[i-1] for i in store_list]
    cs_month_list_for_good_df = [cs_month_list[i-1] for i in store_list]

    year = good_df.Year.values
    month = good_df.Month.values

    good_df['CompetitionTime'] = (year-cs_year_list_for_good_df)*12+(month-cs_month_list_for_good_df)

    p2syear = store_raw.Promo2SinceYear.values
    p2sweek = store_raw.Promo2SinceWeek.values
    woy = good_df.WeekOfYear

    p2syear_for_good_df = [p2syear[i-1] for i in store_list]
    p2sweek_for_good_df = [p2sweek[i-1] for i in store_list]

    good_df['Promo2Time']=(year-p2syear_for_good_df)*12+(woy-p2sweek_for_good_df)

    return good_df

def sale_cus(train,test,store_raw):
    def annual_average(list):
        without_0 = [int(i) for i in list if i!=0]
        return np.mean(without_0)

    train_store_list = train.Store.values

    average_sales_2013 = np.array([annual_average(train[(train.Store==i)&(train.Year==2013)].Sales.values.tolist()) for i in range(1,1116)])
    average_sales_2014 = np.array([annual_average(train[(train.Store==i)&(train.Year==2014)].Sales.values.tolist()) for i in range(1,1116)])
    average_sales_2015 = np.array([annual_average(train[(train.Store==i)&(train.Year==2015)].Sales.values.tolist()) for i in range(1,1116)])

    average_cus_2013 = np.array([annual_average(train[(train.Store==i)&(train.Year==2013)].Customers.values.tolist()) for i in range(1,1116)])
    average_cus_2014 = np.array([annual_average(train[(train.Store==i)&(train.Year==2014)].Customers.values.tolist()) for i in range(1,1116)])
    average_cus_2015 = np.array([annual_average(train[(train.Store==i)&(train.Year==2015)].Customers.values.tolist()) for i in range(1,1116)])

    heat_2013 = average_sales_2013/average_cus_2013
    heat_2014 = average_sales_2014/average_cus_2014
    heat_2015 = average_sales_2015/average_cus_2015

    average_sales_2013_for_train = [average_sales_2013[i-1] for i in train_store_list]
    average_sales_2014_for_train = [average_sales_2014[i-1] for i in train_store_list]
    average_sales_2015_for_train = [average_sales_2015[i-1] for i in train_store_list]

    heat_2013_for_train = [heat_2013[i-1] for i in train_store_list]
    heat_2014_for_train = [heat_2014[i-1] for i in train_store_list]
    heat_2015_for_train = [heat_2015[i-1] for i in train_store_list]


    train['Sales2013']=average_sales_2013_for_train
    train['Sales2014']=average_sales_2014_for_train
    train['Sales2015']=average_sales_2015_for_train

    train['Heat2013']=heat_2013_for_train
    train['Heat2014']=heat_2014_for_train
    train['Heat2015']=heat_2015_for_train

    test_store_list = test.Store.values

    average_sales_2013_for_test = [average_sales_2013[i-1] for i in test_store_list]
    average_sales_2014_for_test = [average_sales_2014[i-1] for i in test_store_list]
    average_sales_2015_for_test = [average_sales_2015[i-1] for i in test_store_list]

    heat_2013_for_test = [heat_2013[i-1] for i in test_store_list]
    heat_2014_for_test = [heat_2014[i-1] for i in test_store_list]
    heat_2015_for_test = [heat_2015[i-1] for i in test_store_list]


    test['Sales2013']=average_sales_2013_for_test
    test['Sales2014']=average_sales_2014_for_test
    test['Sales2015']=average_sales_2015_for_test

    test['Heat2013']=heat_2013_for_test
    test['Heat2014']=heat_2014_for_test
    test['Heat2015']=heat_2015_for_test

    return train,test

