# -*- coding: utf-8 -*-
# coding: utf-8

from xml.etree.ElementInclude import include
import PySimpleGUI as sg
import os.path
import re
import random
import copy
import statistics

#import GPy
#import GPyOpt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time
import downhill_simplex_noFunction as dhs
import epics
from epics import PV, caget, caput

#ここから変更したimport

import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from bayeso_benchmarks import Hartmann6D, Colville
from botorch.settings import validate_input_scaling
import csv
import datetime
import os
import logging
import sys
import threading
import time, datetime
import configparser
import TheSummer.Hitohudebayes as Hitohudebayes
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

#########################################
# 23/10/28
# setting file 書き込み時のバグ修正
# チャージの制限をかけるための機能を追加
# Y settingで読み込んだ値に判定条件を書くことができる
#########################################
# 23/10/24
# 設定ファイル名を画面に残すように改良
###############################################
# 23/03/08
# downhill simplexの機能を追加する．
# downhill_simplex_noFunction.py を使う
######################################
# 22/11/25
# bag 修正
# Init 自動読み込み機能つけた
######################################
# 22/11/17
# y のPVの数を増やす
# デザインも少し変えた
###########################################################
# 22/11/07
# acquisition_weightを指定できるようにした．
###########################################################
# 22/10/28 Takuya Natsui
# setting の save , openを作った．
###########################################################
# 22/10/26 Takuya Natsui
# Epics対応版
# パラメータ変更時のWaitTimeを追加した
# グラフ表示中も計算が止まらないようにした．
###########################################################
###########################################################
# 22/10/20 Takuya Natsui
# GPyOptライブラリを使ってベイズ最適化を行うパネルを作った
# デバック用にmyCaPut, myCaGetを定義したので，
# ここを本当にEPICSを読むように書き換えると動くはず
###########################################################


#---------------------------------------------------------------
#  複数行のpythonで計算できる式をTextで渡して数値を返す
#  最後の式で評価した値を返す
#  返り値はList  [value, bool]
#  評価が成功するとvalue に数値がはいり，bool=True
#  評価が失敗すると valueにはエラー文字列が入り，bool=Falseになる．
#---------------------------------------------------------------
def calc_text(calcStr) :
    lines = calcStr.split('\n')
    last = ""
    for line in lines :
        #print(line)
        try:
            exec(line)
        except ZeroDivisionError:
            print('division by zero')
            return [line + " <- division by zero", False]
        except NameError:
            print('undefined name')
            return [line + " <- undefined name", False]
        
        if line != "" :
            last = line

    try:
        ans = eval(last)
    except ZeroDivisionError:
        print('division by zero')
        return [last + " <- division by zero", False]
    except NameError:
        print('undefined name')
        return [last + " <- undefined name", False]
    except :
        return [last, False]
    else :
        return [ans, True]



# --------------------------------------------------
#  Startが押されたときにUIから値を読むための関数（Push "start" to read value from UI）
# --------------------------------------------------
def getInitSetting( values ) :
    repetition = int(values['rep'])

    dataN = int(values['dataN'])
    iterN = int(values['iterN'])
    WaitTime = float(values['WaitTime'])

    method = 'Baysian'

    UCB = bool(values["UCB"])
    logEI = bool(values["logEI"])
    beta = float(values['beta'])
    
    randomvalue = bool(values["randomvalue"])
    gridvalue = bool(values["gridvalue"])
    bestvalue = bool(values["bestvalue"])
    enqueueData = str(values["enqueueData"])

    if dataN < 1 : dataN = 1
    if dataN > 100 : dataN = 100

    x_name_list = []
    x_min_max_list = []
    x_init_list = []
    x_step_list = []
    x_weight_list = []    
    
    for i in range(0, maxNX) :
        name = values['name_x{}'.format(i)]
        min = values['min_x{}'.format(i)]
        max = values['max_x{}'.format(i)]
        if name != '' and min != '' and max != '':
            x_name_list.append(name)
            x_min_max_list.append( [float(min), float(max)], )
        else :
            break

    for i in range(0, len(x_name_list) ) :
        
        init = values['init_x{}'.format(i)]
        if init != '' :
            x_init_list.append(float(init))
        else :
            x_init_list.append( (x_min_max_list[i][0] + x_min_max_list[i][1])/2 )
            
        step = values['step_x{}'.format(i)]
        if step != '' :
            x_step_list.append(float(step))
        else :
            x_step_list.append(1)
        
        weight = values['weight_x{}'.format(i)]
        if weight != '' :
            x_weight_list.append(float(weight))
        else :
            x_weight_list.append(1)
        

    y_name_list = []
    y_alias_list = []
    for i in range(0, maxNY) :
        name = values['name_y{}'.format(i)]
        alias = values['alias_y{}'.format(i)]
        if name != '' and alias != '' :
            y_name_list.append(name)
            y_alias_list.append(alias)
    
    [y_name_list2, y_alias_list2] = readYsettingText(values)
    y_name_list.extend(y_name_list2)
    y_alias_list.extend(y_alias_list2)
    
    th_name_list = []
    th_alias_list = []
    for i in range(0, maxN_th) :
        name = values['name_th{}'.format(i)]
        alias = values['alias_th{}'.format(i)]
        if name != '' and alias != '' :
            th_name_list.append(name)
            th_alias_list.append(alias)

    limitation = values['limitation']
    
    functionText = values['function']
    if functionText[-1] == '\n' : functionText = functionText[:-1]

    return [repetition, dataN, iterN, WaitTime,
            x_name_list, x_min_max_list, x_init_list, x_step_list, x_weight_list,
            y_name_list, y_alias_list,th_name_list, th_alias_list, limitation, functionText,
            method, UCB, logEI, beta, randomvalue,gridvalue, bestvalue, enqueueData]

#---------------------------------------------------
# Multi TextのエリアからYの設定を読み込む
#---------------------------------------------------
def readYsettingText(values):
    y_name_list2 = []
    y_alias_list2 = []
    text = values['YsettingText']
    pattern = '(\S+)\s+(\S+)\s*'
    lines = text.split('\n')
    for line in lines :
        result = re.match(pattern, line)
        if result :
            y_name_list2.append( result.group(1) )
            y_alias_list2.append( result.group(2) )
    return [y_name_list2, y_alias_list2]



# ----------------------------------------------
#  X name list の Epics Recodeに値をセットする
# ----------------------------------------------
def setValueX_PV(x_name_list,x_step_list ,X , Xold, WaitTime,iter_i, iterN) :
    global log_text
    dx = [0 for i in range(len(x_name_list))]
    nstep = [0 for i in range(len(x_name_list))]
    dstep = [0 for i in range(len(x_name_list))]
    
    for i in range(len(x_name_list)):
        
        dx[i] = X[i] - Xold[i]  # i回目から(i-1)回目の差分
        nstep[i] = int(abs(dx[i]) / x_step_list[i]) + 1  # 差分から、何点を間に挟むか
        dstep[i] = dx[i] / nstep[i]  # 差分を間に挟む点で割った数
    print("X",X)
    print("Xold",Xold)
    print("dx",dx)
    print("nstep",nstep)
    print("dstep",dstep)
    
    for j in range(1, max(nstep) + 1):
        for k in range(len(x_name_list)):
            if j <= nstep[k]:
                caput(x_name_list[k], dstep[k] * j + Xold[k])
                if iter_i < iterN:
                    print( "Iteration {}/{} params x{} split {}/{}".format(iter_i+1, iterN,k,j, nstep[k]))
                    log_text += "Iteration {}/{} params x{} split {}/{}\n".format(iter_i+1, iterN,k,j, nstep[k])
                
                elif iter_i == iterN:
                    print( "Bestparams x{} split {}/{}".format(k,j, nstep[k]))
                    log_text += "Bestparams x{} split {}/{}\n".format(k,j, nstep[k])
                
        time.sleep(WaitTime)
    return

# ----------------------------------------------
#  Y name list の Epics Recodeの値をゲットして
#  functionTextに従って値を計算しそれを返す
# ----------------------------------------------
def getValueY_th_PV(y_name_list, y_alias_list,th_name_list, th_alias_list, functionText) :
    AllText = ""
    for i in range(0, len(y_name_list) ) :
        val = caget(y_name_list[i]) 
        AllText += '{}={}\n'.format(y_alias_list[i], val)
        
    for i in range(0, len(th_name_list) ) :
        val = caget(th_name_list[i]) 
        AllText += '{}={}\n'.format(th_alias_list[i], val)
    AllText += functionText
    calcVal = calc_text(AllText)

    return calcVal

# -----------------------------------------------------------
#   ベイズ最適化の1step
# -----------------------------------------------------------
def optimizationOneStep(study,x_name_list,y_name_list,y_alias_list,x_min_max_list,x_step_list,iter_i,WaitTime,dataN,limitation,functionText) :
    global Xold
    
    log_text = ''
    X = [0 for i in range(len(x_name_list))]
    
    #初期状態から1trial目までをstep by stepでcaputするためにXoldに現在の値を詰める
    if iter_i < 1 :
        Xold = []
        for i in range (len(x_name_list)):
            present_x_val = caget(x_name_list[i])
            Xold.append(present_x_val)
        print(Xold)

    trial = study.ask()

    for i in range(len(x_name_list)):
        X[i] = trial.suggest_float(x_name_list[i], x_min_max_list[i][0], x_min_max_list[i][1])
    
    setValueX_PV(x_name_list,x_step_list ,X , Xold, WaitTime,iter_i, iterN)
    Xold = X.copy()  # Xoldを更新
    
    Y_temp_val_list = []
    Y_temp_val = 0.0
    data_i = 0
    
    while data_i < dataN :
    
        #getValueY_PVの入力がlimitationTextであることに注意。ans = [bool, bool]という値になる
        #最初のboolはlimitationTextが条件式を満たしているか、最後のboolは処理自体が正常に終了しているかを表す。
        if limitation != '' :
            ans = getValueY_th_PV(y_name_list, y_alias_list,th_name_list, th_alias_list, limitation) 
            limit_bool = ans[0]
            #print(limit_bool)
            if not (type(limit_bool) is bool) :
                limit_bool = True
                print(limit_bool)
        else :
            limit_bool = True

        #getValueY_PVの入力がfunctionTextであることに注意。ans = [(functionの計算値), bool]という値になる
        #boolは処理自体が正常に終了しているかを表す。
        if limit_bool == True :
            ans = getValueY_th_PV(y_name_list, y_alias_list,th_name_list, th_alias_list, functionText)
            if ans[1] == False :
                mode = 'stop'
                sg.PopupOK( ans[0] )
            else :
                Y_temp_val = ans[0]
                Y_temp_val_list.append(Y_temp_val)
                print(Y_temp_val_list)
                print('Iteration {}/{}  meas {}/{}  Y_temp_val = {}'.format(iter_i+1, iterN, data_i+1, dataN,Y_temp_val) )
                log_text += 'Iteration {}/{}  meas {}/{}  Y_temp_val = {}\n'.format(iter_i+1, iterN, data_i+1, dataN,Y_temp_val)
                #print(datetime.datetime.now())
                time.sleep(1.1)
                #print(datetime.datetime.now())
            
            window['log'].update(log_text,autoscroll=True)
            data_i += 1

    #Y_val = statistics.mean(Y_temp_val_list)
    Y_val = statistics.median(Y_temp_val_list)
    print('X = {}, Y = {}'.format(X, Y_val))
    log_text += 'new X, Y = {}, {}\n'.format(X, Y_val)
    
    study.tell(trial, Y_val)
    
    return [X,Y_val,log_text]


#-----------------
#グラフ初期化
#-----------------
def setGraph() :
    global ax1
    global ax1_another
    global ax2

    global fig

    fig = plt.figure(figsize=(8,8))

    ax1 = plt.subplot(2,1,1)
    ax1_another = ax1.twinx()
    ax2 = plt.subplot(2,1,2)
    

    return

#-------------------------------
# Graph描画ようY bestの遷移
#-------------------------------
def getBestValues(Y_values_list) :
    val = Y_values_list[0]
    plots = [[0], [Y_values_list[0]]]
    for i in range(0, len(Y_values_list) ) :
        #if val > Y_values_list[i] :
        if val < Y_values_list[i] :
            val = Y_values_list[i]
            plots[0].append(i)
            plots[1].append(val)
    return plots

#------------------------------------------
#  Graphを表示する
#------------------------------------------
def graph_view(X_values_list, Y_values_list, x_min_max_list) :
    # グラフリセット
    ax1.cla()
    ax1_another.cla()
    ax2.cla()

    x_gp_lists = [] #要するにX_values_listの転置行列を作っている
    for i in range(0, len(X_values_list)) :
        for j in range(0, len(X_values_list[i]) ) :
            val = X_values_list[i][j]
            if i == 0 :
                x_gp_lists.append( [ val ] )
            else :
                x_gp_lists[j].append( val )
                
    #for i in range(0, len( x_gp_lists ) ) :
    #    ax1.plot(x_gp_lists[i], label="x{}".format(i))
    ax1.plot(x_gp_lists[0], label="x0", color = "C0")
    ax1.set_ylabel("V-steering_1 (rad)",color = "C0")
    ax1.tick_params(axis = "y", labelcolor = "C0")
    
    plt.text(1.08, 1.71, 'V-steering_2 (rad)', va='center', ha='left', rotation='vertical', 
    transform=plt.gca().transAxes, color='C1')
    ax1_another.plot(x_gp_lists[1],label="x1" ,color = "C1")
    ax1_another.tick_params(labelcolor = "C1")


    Y_BestValues_plot = getBestValues(Y_values_list)
    ax2.plot(Y_values_list, label='Y value')
    ax2.scatter(Y_BestValues_plot[0], Y_BestValues_plot[1], s=30, c='red', marker='o', label='best plot')
    
    #ax1.set_ylabel("V-steering (rad)")
    ax2.set_ylabel("Injection efficiency")
    ax2.set_xlabel("iteration N")

    #ax1.legend() #凡例を表示する場合
    #ax1_another.legend() #凡例を表示する場合
    ax2.legend() #凡例を表示する場合
    ax1.grid() #グリッドを入れる
    ax2.grid() #グリッドを入れる

    #plt.show(block=False) #block=FalseでUIの方を止まらなくする
    #plt.show() #block=Falseは入射器ではできない？

    fig_agg.draw()
    

    return

#-----------------
# 描画用の関数
#-----------------
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


#-----------------------------------------------------
# Setting を保存する
#-----------------------------------------------------
def saveSetting( values, fname ) :

    [repetition, dataN, iterN, WaitTime,
    x_name_list, x_min_max_list, x_init_list, x_step_list, x_weight_list,
    y_name_list, y_alias_list,th_name_list, th_alias_list, limitation, functionText,
    method, UCB, logEI, beta, randomvalue, gridvalue, bestvalue, enqueueData] = getInitSetting( values )

    with open( fname, 'w') as f:
        print('[PV]', file=f)
        print("#Number of tuning parameters", file=f)
        print(f"nxr = {len(x_name_list)}", file=f)
        print("#Number of evaluation parameters", file=f)
        print(f"ny = {len(y_name_list)}", file=f)
        print(f"nth = {len(th_name_list)}", file=f)
        print(f"repetition = {repetition}", file=f)
        print(f"n_trials = {iterN}", file=f)
        print(f"number_of_measurements = {dataN}", file=f)
        print(f"evalsleep = {WaitTime}", file=f)
        print(f"objective_function = {functionText}", file=f)
        print(f"limitation = {limitation}", file=f)
        print('', file=f)

        print("#If the filepath directory does not exist, a new one is created.", file=f)
        print("filepath = ", file=f)
        print(f"source_study = {enqueueData}", file=f)
        print('', file=f)

        print("#Choose one initialization", file=f)
        
        if randomvalue == True:
            print("Initialization = randomvalue", file=f)
            print("#Initialization = gridvalue", file=f)
            print("#Initialization = bestvalue", file=f)
        
        elif gridvalue == True:
            print("#Initialization = randomvalue", file=f)
            print("Initialization = gridvalue", file=f)
            print("#Initialization = bestvalue", file=f)
        
        elif bestvalue == True:
            print("#Initialization = randomvalue", file=f)
            print("#Initialization = gridvalue", file=f)
            print("Initialization = bestvalue", file=f)
        
        else :
            print("#Initialization = randomvalue", file=f)
            print("#Initialization = gridvalue", file=f)
            print("#Initialization = bestvalue", file=f)
        print('', file=f)

        print("#Choose one acquisition function", file=f)
        if logEI == True:
            print("aquisition_function = logEI", file=f)
            print("#aquisition_function = UCB", file=f)
        elif UCB == True:
            print("#aquisition_function = logEI", file=f)
            print("aquisition_function = UCB", file=f)
        else:
            print("#aquisition_function = logEI", file=f)
            print("#aquisition_function = UCB", file=f)
        
        print(f"beta = {beta}", file=f)
        print('', file=f)
        
        for i in range (len(x_name_list)):
            print(f"[PV_XD{i}]", file=f)
            print(f"name = {x_name_list[i]}", file=f)
            print(f"rmin = {x_min_max_list[i][0]}", file=f)
            print(f"rmax = {x_min_max_list[i][1]}", file=f)
            print(f"step = {x_step_list[i]}", file=f)
            print(f"init = {x_init_list[i]}", file=f)
            print(f"weight = {x_weight_list[i]}", file=f)
            print('', file=f)

        for i in range (len(y_name_list)):
            print(f"[PV_Y{i}]", file=f)
            print(f"name = {y_name_list[i]}", file=f)
            print(f"alias = {y_alias_list[i]}", file=f)
            print('', file=f)
        
        for i in range (len(th_name_list)):
            print(f"[PV_th{i}]", file=f)
            print(f"name = {th_name_list[i]}", file=f)
            print(f"alias = {th_alias_list[i]}", file=f)
            print('', file=f)

    return

#-----------------------------------------------------------
# Setting を呼び出す
#-----------------------------------------------------------
def readSetting(window, fname ) :

    config_ini = configparser.ConfigParser()
    config_ini.read(fname)

    xflug = False
    yflug = False
    yTextflug = False
    limitflug = False
    funcflug = False

    nxr = maxNX
    ny = maxNY
    nth = maxN_th
    repetition = 5
    dataN = 3
    iterN = 50
    WaitTime = 1000
    
    x_name_list = []
    x_min_max_list = []
    x_init_list = []
    x_step_list = []
    x_weight_list = []
    y_name_list = []
    y_alias_list = []
    th_name_list = []
    th_alias_list = []
    limitation = ''
    functionText = ''
    YsettingText = ''
    patternX = '(\S+),\s*(\S+),\s*(\S+),\s*(\S+)'
    patternY = '(\S+),\s*(\S+)'

    if os.path.exists(fname) :
        print(fname)
        
        nxr = int(config_ini.get("PV", "nxr"))
        ny = int(config_ini.get("PV", "ny"))
        nth = int(config_ini.get("PV", "nth"))
        repetition = int(config_ini.get("PV", "repetition"))
        dataN = int(config_ini.get("PV", "number_of_measurements"))
        iterN = int(config_ini.get("PV", "n_trials"))
        WaitTime = float(config_ini.get("PV", "evalsleep"))
        limitation = str(config_ini.get("PV", "limitation"))
        functionText = str(config_ini.get("PV", "objective_function"))
        
        enqueueData = str(config_ini.get("PV", "source_study"))
        
        
        if config_ini.get("PV", "Initialization") == "randomvalue":
            randomvalue = True
            gridvalue = False
            bestvalue = False
            
        elif config_ini.get("PV", "Initialization") == "gridvalue":
            randomvalue = False
            gridvalue = True
            bestvalue = False
            
        elif config_ini.get("PV", "Initialization") == "bestvalue":
            randomvalue = False
            gridvalue = False
            bestvalue = True
            
            
        
        else :print("The initialization is not correctly selected.")
        
        if config_ini.get("PV", "aquisition_function") == "logEI":
            logEI = True
            UCB = False
        elif config_ini.get("PV", "aquisition_function") == "UCB":
            logEI = False
            UCB = True
        else : print("The acquisition function is not correctly selected.")
        
        beta = float(config_ini.get("PV", "beta"))
    
        
        for i in range(nxr):
            pvname = config_ini.get("PV_XD{0}".format(i), "name")
            x_name_list.append(pvname)
            rmin = float(config_ini.get("PV_XD{0}".format(i), "rmin"))
            rmax = float(config_ini.get("PV_XD{0}".format(i), "rmax"))
            x_min_max_list.append(list([rmin, rmax]))
            step = float(config_ini.get("PV_XD{0}".format(i), "step"))
            x_step_list.append(step)
            init = float(config_ini.get("PV_XD{0}".format(i), "init"))
            x_init_list.append(init)
            weight = float(config_ini.get("PV_XD{0}".format(i), "weight"))
            x_weight_list.append(weight)
            
        for i in range(ny):
            pvname = config_ini.get("PV_Y{0}".format(i), "name")
            y_name_list.append(pvname)
            alias = config_ini.get("PV_Y{0}".format(i), "alias")
            y_alias_list.append(alias)
            
        for i in range(nth):
            pvname = config_ini.get("PV_th{0}".format(i), "name")
            th_name_list.append(pvname)
            alias = config_ini.get("PV_th{0}".format(i), "alias")
            th_alias_list.append(alias)


        for i in range(0, nxr ) :
            if i < len(x_name_list) :
                window['name_x{}'.format(i)].update(x_name_list[i])
                window['min_x{}'.format(i)].update(x_min_max_list[i][0])
                window['max_x{}'.format(i)].update(x_min_max_list[i][1])
                window['init_x{}'.format(i)].update(x_init_list[i])
                window['step_x{}'.format(i)].update(x_step_list[i])
                window['weight_x{}'.format(i)].update(x_weight_list[i])
                print(x_name_list[i])
                
            else :
                window['name_x{}'.format(i)].update('')
                window['min_x{}'.format(i)].update('')
                window['max_x{}'.format(i)].update('')
                window['init_x{}'.format(i)].update('')
                window['step_x{}'.format(i)].update('')
                window['weight_x{}'.format(i)].update('')


        for i in range(0, ny ) :
            if i < len(y_name_list) :
                window['name_y{}'.format(i)].update(y_name_list[i])
                window['alias_y{}'.format(i)].update(y_alias_list[i])
            else :
                window['name_y{}'.format(i)].update('')
                window['alias_y{}'.format(i)].update('')
                
        for i in range(0, nth ) :
            if i < len(th_name_list) :
                window['name_th{}'.format(i)].update(th_name_list[i])
                window['alias_th{}'.format(i)].update(th_alias_list[i])
            else :
                window['name_y{}'.format(i)].update('')
                window['alias_y{}'.format(i)].update('')

        window['YsettingText'].update(YsettingText[:-1])
        if len(YsettingText) > 5 :
            window['YTextFrame'].update(visible=True)

        window['limitation'].update(limitation)
        window['function'].update(functionText)

        window['rep'].update(repetition)
        window['dataN'].update(dataN)
        window['iterN'].update(iterN)
        window['WaitTime'].update(WaitTime)
        
        window["enqueueData"].update(enqueueData)
        
        window["randomvalue"].update(randomvalue)
        window["gridvalue"].update(gridvalue)
        window["bestvalue"].update(bestvalue)
        
        window["logEI"].update(logEI)
        window["UCB"].update(UCB)
        
        
        
        window['beta'].update(beta)


    return


#-------------------------------------------
# 現在の値を読み取ってInit に入れる
#-------------------------------------------
def SetCurrntValueToInit( values, window, maxNX ) :
    for i in range(0, maxNX ) :
        pvname = values['name_x{}'.format(i)]
        if len(pvname) > 0 :
            val = caget(pvname)
            window['init_x{}'.format(i)].update("{:.3f}".format(val))
            
    return


#-------------------------------------------
# Initの値の周りのMin Maxに設定し直す
#-------------------------------------------
def ShiftMinMaxToInit( values, window, maxNX ) :
    for i in range(0, maxNX ) :
        name = values['name_x{}'.format(i)]
        min = values['min_x{}'.format(i)]
        max = values['max_x{}'.format(i)]
        initVal = values['init_x{}'.format(i)]
        if name != '' and min != '' and max != '' and initVal != '':
            sub = float(max)-float(min)
            val = float(initVal)
            window['min_x{}'.format(i)].update(val-sub/2)
            window['max_x{}'.format(i)].update(val+sub/2)

    return

#-------------------------------------------
# 現在の値を読み取ってInit に入れる
# Initの値の周りのMin Maxに設定し直す
# SetCurrntValueToInit と ShiftMinMaxToInitを連続で行うとvaluesが更新されていないのでうまくいかない．
#-------------------------------------------
def SetCurrntValueToInit_and_ShiftMinMaxToInit( values, window, maxNX ) :
    initVal_list = []
    x_min_max_list = []
    for i in range(0, maxNX ) :
        pvname = values['name_x{}'.format(i)]
        if len(pvname) > 0 :
            initVal = caget(pvname)
            initVal_list.append(initVal)

            window['init_x{}'.format(i)].update("{:.3f}".format(initVal))

            min = values['min_x{}'.format(i)]
            max = values['max_x{}'.format(i)]

            if min != '' and max != '' and initVal != '':
                sub = float(max)-float(min)
                val = float(initVal)
                new_min = val-sub/2
                new_max = val+sub/2
                window['min_x{}'.format(i)].update(new_min)
                window['max_x{}'.format(i)].update(new_max)

                x_min_max_list.append([new_min, new_max])

    return [initVal_list, x_min_max_list]


############################################
# ---------------- main ----------------
############################################
if __name__ == '__main__':

    sg.theme('purple')

    col_Xsetting = [
        [sg.Text('                         PV name', size=(36, 1)),
        sg.Text('min', size=(6, 1)), sg.Text('max', size=(6, 1)),sg.Text('init', size=(5, 1)), sg.Text('step', size=(6, 1)),sg.Text('weight', size=(6, 1))], 
    ]

    maxNX = 6
    for i in range(0, maxNX) :
        col_Xsetting.append( 
            [ sg.Text( 'x{}:'.format(i), size=(2, 1)),
            sg.InputText('', size=(30, 1), key='name_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='min_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='max_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='init_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='step_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='weight_x{}'.format(i) ),
        ])
    col_Xsetting.append(
        [sg.Submit(button_text="Set currnt value to init", key='setInit'), 
            sg.Submit(button_text="Shift MinMax to init", key='shiftMinMax')]
    )


    col_Ysetting = [
        [sg.Text('  PV name', size=(28, 1)),
        sg.Text('alias ', size=(7, 1)),  ], 
    ]

    col_YsettingText = [
        #[sg.Text('Y Setting Text', size=(20, 1)), ], 
        [sg.Multiline("", size=(50, 18), key='YsettingText'), ], 
    ]

    maxNY = 1
    for i in range(0, maxNY) :
        col_Ysetting.append( 
            [ sg.InputText('', size=(30, 1), key='name_y{}'.format(i)),
            sg.InputText('y{}'.format(i), size=(8, 1), key='alias_y{}'.format(i) ),
            ])
    
    #col_Ysetting.append( [sg.Text('Evaluate function : ')] )
    #col_Ysetting.append( [sg.Multiline("", size=(40, 3), key='function') ] )
    
        col_th_setting = [
        [sg.Text('  PV name', size=(28, 1)),
        sg.Text('alias ', size=(7, 1)),  ], 
    ]
        
    maxN_th = 1
    for i in range(0, maxN_th) :
        col_th_setting.append( 
            [ sg.InputText('', size=(30, 1), key='name_th{}'.format(i)),
            sg.InputText('th{}'.format(i), size=(8, 1), key='alias_th{}'.format(i) ),
            ])

    layout1 = [
        [sg.Text("Setting file name :"), sg.InputText('', size=(80, 1), key='SettingFileNameInput'), 
        sg.Submit(button_text="Save", key='saveSettingInputText'), ],
        [sg.Submit(button_text="OpenSetting", key='openSetting'),
        sg.Text("", size=(2,1) ),
        sg.Submit(button_text="SaveSetting", key='saveSetting'), 
        sg.Text("", size=(15,1) ), sg.Text("Y setting text: ", size=(10,1) ),
        sg.Submit(button_text="ON", key='YTextOn'), sg.Submit(button_text="OFF", key='YTextOff'),], 
        [sg.Frame('X settings', [[sg.Column(col_Xsetting)]]),
        sg.Column([
        [sg.Frame('Y settings', [[sg.Column(col_Ysetting)]])],
        [sg.Frame('Y settings Text', [[sg.Column(col_YsettingText)]], visible=False, key='YTextFrame')],
        [sg.Frame('Threshold settings', [[sg.Column(col_th_setting)]])]
        ])],
        [sg.Text("Limitation :"), sg.InputText('', size=(80, 1), key='limitation'), ],#元のコード
        #[sg.Text('Evaluate function : '), sg.Multiline("", size=(80, 4), key='function') ],#元のコード
        [sg.Multiline("", size=(80, 4),visible = False , key='function') ],#windowからは見えなくしたコード
        [sg.Text("Beam repetition:"), sg.InputText(5, size=(7 ,1), key='rep'), sg.Text("Hz  ") , 
        sg.Text(" data N at a point:"), sg.InputText(3, size=(7 ,1), key='dataN'), 
        sg.Text(" Iteration N :"), sg.InputText(50, size=(7 ,1), key='iterN'),
        sg.Text(" Wait Time [sec]:"), sg.InputText(1, size=(7 ,1), key='WaitTime'), ], 
        [ sg.Frame( ' Acquisition function ', [
        [sg.Radio('  UCB   ',  key='UCB', group_id='0', default=True), 
        sg.Text("beta:"), sg.InputText(1, size=(5 ,1), key='beta'), sg.Text("defalt:2, exploration:3")],
        [sg.Radio('   EI    ', key='logEI', group_id='0')]]),
        sg.Frame( ' Initialization ', [
        [sg.Text("                                                             referenced data")],
        [sg.Radio('  Random   ',  key='randomvalue', group_id='1', default=True), 
        sg.Radio('  Grid   ',  key='gridvalue', group_id='1'),
        sg.Radio('  Best value   ',  key='bestvalue', group_id='1'), sg.InputText("input enqueue data", size=(20 ,1), key='enqueueData')]])],
        [sg.Submit(button_text="   Start   ", key='start', font=('Arial', 16) ), sg.Checkbox("with set current and shift", default=False, key = "setCurrntShift"), 
        sg.Text("", size=(2,1) ),  
        sg.Submit(button_text="  Stop  ", key='stop'),
        sg.Submit(button_text="  Restart  ", key='restart'),
        sg.Submit(button_text=" Set Best and Finish  ", key='setBestFinish'),
        sg.Text("", size=(10,1) ),
        #sg.Submit(button_text="  Abort  ", key='abort', button_color=('white', 'red')), 
        ],
        [sg.Text("stop", size=(100 ,1), key='info'),],

        [ sg.Multiline("", size=(122, 18), key='log')],
    ]
    
    layoutGraph = [
        [sg.Canvas(key='-CANVAS-')], 
        [sg.Text("stop", size=(100 ,1), key='info2'),],
    ]

    layout = [
        [sg.TabGroup([[
            sg.Tab('Optimaize', layout1),
            sg.Tab('Graph', layoutGraph)
        ]])],
    ]
    

    window = sg.Window("Gereral Optimizer UI", layout, finalize=True, )
    #window = sg.Window("Gereral Optimizer UI", layout, )

    setGraph()
    # figとCanvasを関連付ける．
    fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    t = 500

    # ----------   Example Input  -------------
    event, values = window.read(timeout=t,timeout_key='-timeout-')
    window['name_x0'].update('TEST:X0')
    window['min_x0'].update(-10)
    window['max_x0'].update(10)
    window['init_x0'].update(0)
    window['step_x0'].update(5)
    window['weight_x0'].update(1)
    window['name_x1'].update('TEST:X1')
    window['min_x1'].update(-10)
    window['max_x1'].update(10)
    window['init_x1'].update(0)
    window['step_x1'].update(5)
    window['weight_x1'].update(1)
    #window['name_x2'].update('TEST:X2')
    #window['min_x2'].update(-10)
    #window['max_x2'].update(10)
    #window['init_x2'].update(0)
    #window['step_x2'].update(5)
    #window['weight_x2'].update(1)
    #window['name_x3'].update('TEST:X3')
    #window['min_x3'].update(-10)
    #window['max_x3'].update(10)
    #window['init_x3'].update(0)
    #window['step_x3'].update(5)
    #window['weight_x3'].update(1)

    window['name_y0'].update('TEST:Y')
    window['function'].update('y0')
    
    window['name_th0'].update('TEST:X0')
    window['function'].update('th0')
    
    
    
    

    #------------------------------------------


    # 自動制御のシーケンスは
    # meas phase でdataN回だけ値を測定しcalcに移行する
    # calc phase で次の測定点をきめる．
    # iterN回が終わったらstop modeにうつる．
    mode = 'stop' # or run,

    # counter
    iter_i = 0
    data_i = 0

    # log text
    log_text = ""

    X_values_list = [] #2次元list
    newX = []
    Y_values_list = []

    bestY_value = 9999
    bestXset = []

    while True: #ループに入る
        event, values = window.read(timeout=t,timeout_key='-timeout-') #timeoutの単位はms
        
        if event is None:
            print('exit')
            break
        
        if event == 'openSetting':
            fname = sg.popup_get_file('open file', file_types=(("text Files", ".ini"), ("all Files", "*.*")) )
            if type(fname) is str and len(fname)>0 :
                readSetting(window, fname )
                window['SettingFileNameInput'].update(fname)
        if event == 'saveSetting':
            fname = sg.popup_get_file('save as', save_as=True, file_types=(("text Files", ".ini"), ("all Files", "*.*")) )
            if type(fname) is str and len(fname)>0 :
                saveSetting( values, fname )
                window['SettingFileNameInput'].update(fname)
        if event == 'saveSettingInputText':
            fname = values['SettingFileNameInput']
            if type(fname) is str and len(fname)>0 :
                print(f'fname:{fname}')
                if os.path.isfile(fname):
                    if "Yes" == sg.PopupYesNo( f"Over write? : {fname}" )  :
                        saveSetting( values, fname )
                else :
                    saveSetting( values, fname )


        if event == 'YTextOn':
            window['YTextFrame'].update(visible=True)
        if event == 'YTextOff':
            window['YTextFrame'].update(visible=False)

        if event == 'setInit' :
            SetCurrntValueToInit(values, window, maxNX)

        if event == 'shiftMinMax' :
            ShiftMinMaxToInit(values, window, maxNX)


        if event == "start":
            print("start")

            log_text = ""
            
            data_i = 0
            iter_i = 0

            X_values_list = [] #2次元list
            Y_values_list = []

            bestY_value = 0
            bestXset = []
            log_text += "start\n" 

            [repetition, dataN, iterN, WaitTime,
            x_name_list, x_min_max_list, x_init_list, x_step_list, x_weight_list,
            y_name_list, y_alias_list,th_name_list, th_alias_list, limitation, functionText,
            method, UCB, logEI, beta, randomvalue,gridvalue, bestvalue, enqueueData] = getInitSetting( values )
            

            if values["setCurrntShift"] :
                #"with set current and shift" にチェックが入っていたら現在値を読み込んで範囲もシフト
                [x_init_list, x_min_max_list] = SetCurrntValueToInit_and_ShiftMinMaxToInit( values, window, maxNX )


            #if repetition < 1 : repetition = 1
            if repetition > 50 : repetition = 50
            t = 1000/repetition #timeoutして自動更新する時間を決める

            newX = [] 

            log_text += "Repetition {}, dataN {}, IterN {} \n".format(repetition, dataN, iterN,)
            print(x_name_list)
            print(x_min_max_list)
            print(x_init_list)
            print(y_name_list)
            print(y_alias_list)
            print(functionText)

            window['log'].update(log_text)
            window['info'].update(f'run {method}')
            window['info2'].update(f'run {method}')
            
            x_init_dict = {}
            for i in range (len(x_name_list)):
                x_init_dict[f"{x_name_list[i]}"] = x_init_list[i]
                
            #------------------------------------------
            #ここから自作関数(startが押されたらstudyを立ち上げる)
            #------------------------------------------
            
            now = datetime.datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            
            # --- Objective function ---
            study = optuna.create_study(
                # sampler=optuna.samplers.TPESampler(),
                # sampler=optuna.samplers.CmaEsSampler(source_trials=source_study.trials),
                # sampler=optuna.samplers.CmaEsSampler(),
                #direction = "maximize",
                direction = "minimize",
                # sampler=optuna.integration.BoTorchSampler(),
                study_name="{}".format(current_time),
                storage="sqlite:///SKEKB20240208.db",
                sampler=Hitohudebayes.HitohudebayesSampler(x_name_list,x_min_max_list,x_weight_list,UCB,logEI,beta,n_startup_trials = 4)
            )
            
            if randomvalue == True:
                study.enqueue_trial(x_init_dict)
                print(x_init_dict)
            
            elif gridvalue == True:
                #study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+x_min_max_list[0][1])/2,f"{x_name_list[1]}" : (x_min_max_list[1][0]+x_min_max_list[1][1])/2})
                study.enqueue_trial({f"{x_name_list[0]}" : (3*x_min_max_list[0][0]+x_min_max_list[0][1])/4,f"{x_name_list[1]}" : (x_min_max_list[1][0]+x_min_max_list[1][1])/2})
                study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+3*x_min_max_list[0][1])/4,f"{x_name_list[1]}" : (x_min_max_list[1][0]+x_min_max_list[1][1])/2})
                study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+x_min_max_list[0][1])/2,f"{x_name_list[1]}" : (3*x_min_max_list[1][0]+x_min_max_list[1][1])/4})
                study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+x_min_max_list[0][1])/2,f"{x_name_list[1]}" : (x_min_max_list[1][0]+3*x_min_max_list[1][1])/4})
            
            elif bestvalue == True:
                source_study = optuna.load_study(
                study_name = enqueueData,
                storage="sqlite:///SKEKB20240208.db"
            )
                # 最大化
                for trial in sorted(source_study.trials, key=lambda t: t.value)[90:]:
                
                # 最小化
                #for trial in sorted(source_study.trials, key=lambda t: t.value)[:10]:
                    study.enqueue_trial(trial.params)
                    print(trial.params)
                    
                
            else : print("The acquisition function is not correctly selected.")
            
            filename = "./log_" + current_time + ".csv"
                
            #------------------------------------------
            #ここまで自作関数(その後runに切り替える)
            #------------------------------------------
            mode = 'run'

        if event == "stop":
            log_text += 'Iteration {}/{}  meas {}/{} : best y = {} at x = {}\n'.format(
            iter_i+1, iterN, data_i+1, dataN, bestY_value, bestXset)
            window['log'].update(log_text)
            window['info'].update('stop')
            window['info2'].update('stop')
            mode = 'stop'

        if event == "restart":
            mode = 'run'

        if event == 'setBestFinish' :
            log_text += 'Iteration {}/{}  meas {}/{} : best y = {} at x = {}\n'.format(
            iter_i+1, iterN, data_i+1, dataN, bestY_value, bestXset)
            window['log'].update(log_text)
            window['info'].update('stop')
            window['info2'].update('stop')

            data_i = 0
            iter_i = 0
            mode = 'stop'

            if len(bestXset) > 1 :
                setValueX_PV(x_name_list, bestXset) #最適値をセットする


        #if event == "abort":
        #    data_i = 0
        #    iter_i = 0
        #    mode = 'stop'

        if event == 'graph' :
            graph_view(X_values_list, Y_values_list, x_min_max_list)

        if mode == 'run' :

            #if values['UCB'] : acqf_name = 'UCB'
            #elif values['logEI'] : acqf_name = 'EI'
            
            if method == 'Baysian' : #------- Baysian optimaze ----------------
                
                [X_vals,Y_val,optimization_text] = optimizationOneStep(study,x_name_list,y_name_list,y_alias_list,x_min_max_list,x_step_list,iter_i,WaitTime,dataN,limitation,functionText)
                
                X_values_list.append(X_vals)
                Y_values_list.append(Y_val)
                log_text += optimization_text
                
                #if Y_val < bestY_value or iter_i == 0 :
                if Y_val > bestY_value or iter_i == 0 :
                    bestY_value = Y_val
                    bestXset = X_vals
                
                with open(filename,"a") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                    [iter_i, Y_val, study.best_value] + X_vals + list(study.best_params.values())
                    )
                    fig.savefig("./GUIplot" + current_time + ".png")
                
                info_text = f"best y = {study.best_value} at x = {list(study.best_params.values())}"
                log_text += info_text +'\n'
                
                window['info'].update(info_text)
                window['info2'].update(info_text)
                
                graph_view(X_values_list, Y_values_list, x_min_max_list)
                window['log'].update(log_text)

                iter_i += 1

                if iter_i == iterN :
                    mode = 'stop'
                    graph_view(X_values_list, Y_values_list, x_min_max_list)
                    setValueX_PV(x_name_list,x_step_list ,list(study.best_params.values()) , Xold, WaitTime,iter_i, iterN) #最後に最適値をセットする
                    print(f"best y = {study.best_value} at x = {list(study.best_params.values())}")
                    print('Finish')
                    log_text += 'Finish\n'
                    data_i = 0
                    iter_i = 0
                    window['log'].update(log_text)
                    
                    fig.savefig("./GUIplot" + current_time + ".png")
