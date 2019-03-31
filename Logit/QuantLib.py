# -*- coding:utf-8 -*-
from __future__ import division      #除数可以显示为float

from six import StringIO    #使用聚宽readfile函数


import time                 #使用time stamp
import datetime             

import matplotlib.pyplot as plt

import math

import talib

import numpy as np
import pandas as pd

# 功能：从聚宽函数库中提取数据
# 输入：
# stock_list_all：需要生成股票的列表
# sample_count：生成的样本个数
# terminal_date：终止日期
# win_num：胜率
# lose_num：败率
# even_num：最长持仓天数
# 输出
# pandas数据，包含'code','date','win_rate','close','high','low','volume','open'。
def get_data(stock_list_all,sample_count,terminal_date,win_num,lose_num,even_num):
    
    # 总的统计数据，用于汇总所有
    Data_Total_pd = pd.DataFrame(columns  = ['code','date','win_rate','close','high','low','volume','open'])
    
    ##########################1、生成第一列：价格     #############################################
    for stock_name in set(stock_list_all):
        # 按照每个股票循环的要求，每次循环对数据进行清零

        pd_price = pd.DataFrame(columns  = ['code','date','win_rate','close','high','low','volume','open'])


        # 一、开始取数据
        stock_price = get_price(stock_name, count = sample_count, end_date=terminal_date, frequency='1d', 
                                    fields=['open','high','low','close','volume'], fq='none')


        # 二、开始对pd_price赋值
        # 取收盘价作为参考

        pd_price['code'] =  np.array([stock_name for i in range(0,len(list(stock_price['close'])))])
        pd_price['date'] = np.array(stock_price.index)

        pd_price['close'] =  np.array(stock_price['close'])
        pd_price['high'] =  np.array(stock_price['high']) 
        pd_price['low'] =  np.array(stock_price['low'])    
        pd_price['volume'] =  np.array(stock_price['volume'])    
        pd_price['open'] =  np.array(stock_price['open'])    

        
    ############################## 2、生成第二列，每日的胜败关系 ##########################################################

        # 获取价格信息
        price_stock = []
        price_stock = list(stock_price.loc[:,'close'])

        # 记录胜负信息
        win_rate_stock = []

        # 开始针对该股票的价格进行胜负判断
        # 判断标准
        # 1、大于win_num记为“win”
        # 2、小于lose_num记为“lose”
        # 3、多过even_num天记为“even”
        # 设置序号，遍历整个价格数组
        # 注意：range函数生成的序列不包含函数中的第二个参数（也就是最后一个数）
        
        # 第一次遍历前半部分
        for i in range(0,int(len(price_stock)-even_num)):
            for day_count in range(1,even_num+1):
                if price_stock[i+day_count] / price_stock[i] >= win_num:
                    win_rate_stock.append(float(1))
                    break
                if price_stock[i+day_count] / price_stock[i] <= lose_num:                
                    win_rate_stock.append(float(-1))
                    break
                if day_count >= even_num:
                    win_rate_stock.append(float(0))
                    break

        # 第二次遍历even_num后半部分 
        for i in range(int(len(price_stock)-even_num),int(len(price_stock))):
            win_rate_stock.append(float(0))

        # 开始给pandas函数赋值
        pd_price.loc[pd_price['code'] == stock_name,"win_rate"] = np.array(win_rate_stock[:])
    
        pd_price = pd_price.fillna(0)

    ############################## 3、拼接数据 #############################################################
        # for 循环内拼接数据
        Data_Total_pd = pd.concat([Data_Total_pd,pd_price])

    return Data_Total_pd

# 功能：从数据中提取价格生成胜负关系
# 输入：
# input_data_pd：输入的pandas数据
# win_num：胜率
# lose_num：败率
# even_num：最长持仓天数
# 输出
# pandas数据，包含'win_rate'
def Win_rate_Generate(input_data_pd,win_num,lose_num,even_num):
    
    Data_Total_pd = input_data_pd.copy()
    Data_Total_pd ['win_rate'] = None
    
    stock_list = set(list(Data_Total_pd['code']))
    
    for stock_name in stock_list:
        # 获取价格信息
        price_stock = []
        price_stock = list(Data_Total_pd.loc[Data_Total_pd['code'] == stock_name,'close'])

        # 记录胜负信息
        win_rate_stock = []

        # 开始针对该股票的价格进行胜负判断
        # 判断标准
        # 1、大于win_num记为“win”
        # 2、小于lose_num记为“lose”
        # 3、多过even_num天记为“even”
        # 设置序号，遍历整个价格数组
        # 注意：range函数生成的序列不包含函数中的第二个参数（也就是最后一个数）
        
        # 第一次遍历前半部分
        for i in range(0,int(len(price_stock)-even_num)):
            for day_count in range(1,even_num+1):
                if price_stock[i+day_count] / price_stock[i] >= win_num:
                    win_rate_stock.append(float(1))
                    break
                if price_stock[i+day_count] / price_stock[i] <= lose_num:                
                    win_rate_stock.append(float(-1))
                    break
                if day_count >= even_num:
                    win_rate_stock.append(float(0))
                    break

        # 第二次遍历even_num后半部分 
        for i in range(int(len(price_stock)-even_num),int(len(price_stock))):
            win_rate_stock.append(float(0))

        # 开始给pandas函数赋值
        Data_Total_pd.loc[Data_Total_pd['code'] == stock_name,"win_rate"] = np.array(win_rate_stock[:])
    
    
    Data_Total_pd = Data_Total_pd.fillna(0)
    return Data_Total_pd    
    

# 功能：输入Pandas数据，生成对应的指标数据
# 输入：
# input_data_pd：输入的原始Pandas数据
# 格式：必须包含的列包括：'code','date','close','high','low','volume','open'，用于计算指标
# 输出：
# ret_pd：输出Pandas数据
# ret_col：返回的特征向量
# 计算 15个指标，用于计算的指标包括：
# 'EMA_5', 'EMA_10', 'EMA_gap', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'RSI', 'MACD_dif', 'MACD_dea', 'MACD_macd','MOM_12', 'MOM_25', 'MOM_gap', 'Long_Short_Rate_OBV', 'Volume'

def Indicator_Generate(input_data_pd):
    
    ret_pd = pd.DataFrame()
    ret_pd = input_data_pd
    
    # 返回的特征向量
    ret_col = ['EMA_5', 'EMA_10', 'EMA_gap', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'RSI', 'MACD_dif', 'MACD_dea', 'MACD_macd','MOM_12', 'MOM_25', 'MOM_gap', 'Long_Short_Rate_OBV', 'Volume','CCI','SAR']
    
    stock_list = set(list(input_data_pd['code']))
    
    # 计算相应的值
    for stock_name in stock_list:
    
        ################################   1、生成MACD信息   ###################################################
        ret_pd["MACD_dif"] = None
        ret_pd["MACD_dea"] = None
        ret_pd["MACD_macd"] = None
        
        for stock_name in stock_list:
            dif = []
            dea = []
            macd = []

            macd_price = ret_pd.loc[ret_pd['code'] == stock_name,"close"]

            dif, dea, macd = talib.MACD(np.array(macd_price), fastperiod=5, slowperiod=10, signalperiod=5)

            ret_pd.loc[ret_pd['code'] == stock_name,"MACD_dif"] = np.array(dif)
            ret_pd.loc[ret_pd['code'] == stock_name,"MACD_dea"] = np.array(dea)
            ret_pd.loc[ret_pd['code'] == stock_name,"MACD_macd"] = np.array(macd)


        ################################   2、生成均线信息   ###################################################
        ret_pd["EMA_5"] = None
        ret_pd["EMA_10"] = None
        ret_pd["EMA_gap"] = None

        
        
        for stock_name in stock_list:
            ema_5 = []
            ema_10 = []
            ema_gap = []


            ema_price = ret_pd.loc[ret_pd['code'] == stock_name,"close"]
            ema_5 = talib.EMA(np.array(ema_price), timeperiod=5)
            ema_10 = talib.EMA(np.array(ema_price), timeperiod=10)
            ema_gap = talib.EMA(np.array(ema_price), timeperiod=5) - talib.EMA(np.array(ema_price), timeperiod=10)


            ret_pd.loc[ret_pd['code'] == stock_name,"EMA_5"] = np.array(ema_5)
            ret_pd.loc[ret_pd['code'] == stock_name,"EMA_10"] = np.array(ema_10)
            ret_pd.loc[ret_pd['code'] == stock_name,"EMA_gap"] = np.array(ema_gap)


        ################################   3、生成KDJ指标   ###################################################
        ret_pd["KDJ_K"] = None
        ret_pd["KDJ_D"] = None    
        ret_pd["KDJ_J"] = None
        
        
        
        for stock_name in set(stock_list):
            K_values = []
            D_values = []
            J_values = []

            kdj_price = ret_pd.loc[ret_pd['code'] == stock_name,['high','low','close']]
            kdj_price = kdj_price.fillna(0)

                #                  (Today's Close - LowestLow)
                # FASTK(Kperiod) = --------------------------- * 100
                #                   (HighestHigh - LowestLow)

                # FASTD(FastDperiod) = MA Smoothed FASTK over FastDperiod

                # SLOWK(SlowKperiod) = MA Smoothed FASTK over SlowKperiod

                # SLOWD(SlowDperiod) = MA Smoothed SLOWK over SlowDperiod

            K_values, D_values = talib.STOCH(kdj_price['high'].values,
                                               kdj_price['low'].values,
                                               kdj_price['close'].values,
                                               fastk_period=9,
                                               slowk_period=3,
                                               slowk_matype=0,
                                               slowd_period=3,
                                               slowd_matype=0)


            J_values = 3 * np.array(K_values) - 2 * np.array(D_values)

            ret_pd.loc[ret_pd['code'] == stock_name,"KDJ_K"] = np.array(K_values)
            ret_pd.loc[ret_pd['code'] == stock_name,"KDJ_D"] = np.array(D_values)
            ret_pd.loc[ret_pd['code'] == stock_name,"KDJ_J"] = np.array(J_values)


        ################################   4、开始计算RSI指标   ###################################################
        ret_pd["RSI"] = None

        
        for stock_name in set(stock_list):
            rsi_values = []

            rsi_price = ret_pd.loc[ret_pd['code'] == stock_name,['high','low','close']]

            rsi_price = rsi_price.fillna(0)

            rsi_values = talib.RSI(np.array(rsi_price['close']), 12)       #RSI的天数一般是6、12、24

            ret_pd.loc[ret_pd['code'] == stock_name,"RSI"] = np.array(rsi_values)

        ret_pd = ret_pd.fillna(0)

        ################################   5、开始计算动量指标“MOM”   ###################################################
        ret_pd["MOM_12"] = None
        ret_pd["MOM_25"] = None
        # 选择一条10日均线作为中间线，判断
        ret_pd["MOM_gap"] = None

        
        for stock_name in set(stock_list):

            MOM_values = []
            MOM_gap_values = []

            mom_price = list(ret_pd[ret_pd['code'] == stock_name]['close'])


            MOM_12_values = talib.MOM(np.array(mom_price), timeperiod = 12)

            MOM_25_values = talib.MOM(np.array(mom_price), timeperiod = 25)

            MOM_gap_values = MOM_25_values - MOM_12_values

            ret_pd.loc[ret_pd['code'] == stock_name,"MOM_12"] = np.array(MOM_12_values)
            ret_pd.loc[ret_pd['code'] == stock_name,"MOM_25"] = np.array(MOM_25_values)

            ret_pd.loc[ret_pd['code'] == stock_name,"MOM_gap"] = np.array(MOM_gap_values)


        ################################   6、开始计算能量指标OBV   ###################################################
        ret_pd["Long_Short_Rate_OBV"] = None


        
        for stock_name in set(stock_list):
            OBV_values = []


            obv_price = ret_pd.loc[ret_pd['code'] == stock_name,['high','low','close','volume']]

            obv_price = obv_price.fillna(0)

            # 通过价格进行调整
            Long_Short_Rate_OBV_values = []


            OBV_gap = 0 
            for i in range(0,len(obv_price['close'])):
                OBV_gap = ((obv_price['close'].values[i]-obv_price['low'].values[i]) \
                               -(obv_price['high'].values[i]-obv_price['close'].values[i])) \
                              /(obv_price['high'].values[i]-obv_price['low'].values[i])


                if np.isnan(OBV_gap):
                    OBV_gap = 0

                if i == 0:
                    Long_Short_Rate_OBV_values.append(obv_price['volume'].values[i])
                else:
                    Long_Short_Rate_OBV_values.append(float(OBV_gap)*float(obv_price['volume'].values[i]))

            ret_pd.loc[ret_pd['code'] == stock_name,"Long_Short_Rate_OBV"] = np.array(Long_Short_Rate_OBV_values)       


        ################################  7、开始计算成交量   ###################################################
        ret_pd["Volume"] = None

        for stock_name in set(stock_list):

            Volume_price = ret_pd.loc[ret_pd['code'] == stock_name,['high','low','close','volume']]

            ret_pd.loc[ret_pd['code'] == stock_name,"Volume"] = np.array(Volume_price['volume'])       

            
            
        ################################  8、开始计算CCI   ###################################################
        ret_pd["CCI"] = None

        for stock_name in set(stock_list):
            
            CCI_values = []
            
            CCI_price = ret_pd.loc[ret_pd['code'] == stock_name,['high','low','close','volume']]

            CCI_values = talib.CCI(np.array(CCI_price['high']), np.array(CCI_price['low']),np.array(CCI_price['close']), timeperiod=14)
            
            ret_pd.loc[ret_pd['code'] == stock_name,"CCI"] = np.array(CCI_values)       

            
        ################################  9、开始计算SAR   ###################################################
        ret_pd["SAR"] = None

        for stock_name in set(stock_list):
            
            SAR_values = []
            
            SAR_price = ret_pd.loc[ret_pd['code'] == stock_name,['high','low','close','volume']]

            SAR_values = talib.SAR(np.array(SAR_price['high']), \
                                   np.array(SAR_price['low']), \
                                    acceleration=0, maximum=0)
            
            ret_pd.loc[ret_pd['code'] == stock_name,"SAR"] = np.array(SAR_values)       

            
        ret_pd = ret_pd.fillna(0)
    
    return ret_pd,ret_col



# 功能：数据数据后，按照日期，进行相应的扩展，将前几天的数据进行扩展，分别显示前N天的数据
# 输入：
# input_data_pd：输入的原始Pandas数据，函数中必须包含'code'，'date','win_rate'
# day_count：需要向前扩展的天数
# Col_Name：需要扩展的列，避免将'code'、'date','win_rate'也同步扩展了
# 输出：
# ret_pd：输出扩展后的Pandas数据
# Col_Total:输出扩展后的列表列名
def Indicator_Extend(input_data_pd,day_count,Col_Name):

    # 生成总的列表
    Col_Total = []
    for col in Col_Name:
        Col_Total.append(col)
        
        # 生成前两天的值
        for day_count_i in range(1, day_count+1):
            Col_Total.append(str(str(col) + "_pre" + str(day_count_i)))
    
    
    
    # 初始化数据
    ret_pd = pd.DataFrame(columns = (['code','date'] +Col_Total))    
    ret_pd[['code','date','win_rate']] =  input_data_pd.loc[:,['code','date','win_rate']]
    
    ret_pd[Col_Name] =  input_data_pd.loc[:,Col_Name]
    Stock_list = set(list(input_data_pd['code']))
    
    ##############################################  1、 赋值前1-2天的值      ############################################################

    for col in Col_Name:

        # 修改列属性为float,之后数据透视需要
        ret_pd[col] = ret_pd[col].astype(float)

        for day_count_i in range(1, day_count + 1):

            ret_pd.loc[:, str(str(col) + "_pre" + str(day_count_i))] = None

            # 修改列属性为float,之后数据透视需要
            ret_pd[str(str(col) + "_pre" + str(day_count_i))] = ret_pd[str(str(col) + "_pre" + str(day_count_i))].astype(float)

            for stock_name in Stock_list:
                temp = []
                temp = list(input_data_pd.loc[input_data_pd['code'] == stock_name, col])
                for i in range(0, day_count_i):
                    temp.insert(0,0)

                
                ret_pd.loc[input_data_pd['code'] == stock_name, str(str(col) + "_pre" + str(day_count_i))] = np.array(temp[:-day_count_i])


    return ret_pd,Col_Total


# 功能：删除不必要的数据，包括
# 1、前60个数据
# 2、后10个数据
# 3、成交量为0的数据
# 返回：
# 删除完成后的数据
def delete_data(input_data_pd):

    ret_pd = input_data_pd.copy()
    ret_pd.index = range(0,ret_pd.shape[0])
    
    # 删除无效数据，准备进行数据处理
    stock_list = set(list(ret_pd['code']))
    # 更改平局为-1
    ret_pd.loc[ret_pd['win_rate'] == 0, "win_rate"] = -1
    for stock_name in stock_list:
        # 删除前60，后10的数据
        ret_pd = ret_pd.drop(ret_pd[ret_pd['code'] == stock_name].iloc[:60].index, axis=0)
        ret_pd = ret_pd.drop(ret_pd[ret_pd['code'] == stock_name].iloc[-10:].index, axis=0)
        # 统一操作，删除价格为0的数据
        ret_pd = ret_pd.drop(ret_pd[ret_pd['Volume'] == 0].index, axis=0)
    return ret_pd



# normalize_into_pd
# 功能：将数据进行归一化，逐股票生成相应的离散值
# 离散原则：按照正态分布的规则，按照均值、方差的值，分阶段进行离散
# input_data_pd：pandas数据结构，多行数据，记录样本归一化前的原始数据
# cols：列名，归一化的相应列名
# 返回值：
# ret_pd：按照原则离散化后的Pandas数组
# MS_pd：返回按照股票、cols列为维度的Pandas数据

def normalize_into_pd(input_data_pd,cols):
    # 定义分组函数，对输入的数列按照n组进行划分
    # 输入数据为list类型
    def get_group(sample_data):
        # 返回结果的数组
        ret_group = []
        if len(sample_data)!=0:
            # 确定最大值、最小值
            # 同时gap适当扩大，避免出现n+1的分组
            d_mean = np.mean(sample_data)
            d_std = np.std(sample_data)
            if d_std != 0:
                ret_group = [math.floor(vals) if abs(math.floor(vals)) <=3 else 4*abs(math.floor(vals))/math.floor(vals) for vals in (np.array(sample_data) - d_mean)/(0.5*d_std)]
            else:
                ret_group = [0  for vals in range(0,len(sample_data))]
        else:
            ret_group =  []
        return ret_group
    
    # 变量赋初值，其中input_data_pd必须包含code列
    Stock_list = set(list(input_data_pd['code']))
    ret_pd = pd.DataFrame(columns = ['code','date'] + cols)
    MS_pd = pd.DataFrame()

    ret_pd['code'] = np.array(input_data_pd['code'])
    ret_pd['date'] = np.array(input_data_pd['date'])
    ret_pd['win_rate'] = np.array(input_data_pd['win_rate'])
    
    # 逐列开始进行数据处理
    for col in cols:

        # 处理原始列数据

        for stock_name in Stock_list:
            ret_pd.loc[ret_pd['code'] == stock_name, col] \
                = np.array(get_group(list(input_data_pd[input_data_pd['code'] == stock_name][col])))
    
        
    MS_pd = pd.pivot_table(input_data_pd, index=["code"], values=cols, aggfunc=[np.mean,np.std])

    return ret_pd,MS_pd


# normalize_by_list使用索引表对数据进行归一化，生成模型可识别的样本
# file_name：索引文件名，记录均值方差，需遵守相应表格列名称规则
# sample_data_pd：pandas数据结构，仅一行，记录样本归一化前的原始数据
# stock_name：股票名，用于去索引文件中查找对应的数据
# cols：列名，归一化的相应列名

def normalize_by_list(MS_pd,sample_data_series,stock_name,cols):
    # ret_pd是返回的pd数组
    ret_pd = pd.DataFrame(columns = cols)
    
    # 创建列，赋值为None
    ret_pd[cols] = None
    
       
    # 逐列计算对应归一化的值
    for col in cols:
        mean = 0
        std = 0
        
        # 查找表中的数据
        mean = MS_pd.loc[MS_pd['code']==stock_name,str(col+"_mean")]
        std = MS_pd.loc[MS_pd['code']==stock_name,str(col+"_std")]

        # 计算
        if std !=0 :
            val = (float(sample_data_series[col]) - mean)/(0.5*std)

            temp = 0
            if abs(math.floor(val)) <=3:
                temp = math.floor(val)
            else:
                temp = 4*val/abs(val)
        else:
            temp = 0
        ret_pd.loc[:,col] = np.array([float(temp)])

    return ret_pd


