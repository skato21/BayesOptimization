# -*- coding: utf-8 -*-
# coding: utf-8

from xml.etree.ElementInclude import include
import PySimpleGUI as sg
import os.path
import re
import random
import copy


#import GPy
#import GPyOpt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import random

import time

import downhill_simplex_noFunction as dhs

import epics
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

#---------------------------------------------
#  caget, caputの部分
#  まずはデバッグ用の関数になっているが，ここを変えれば実機で使用できるようになる．
#---------------------------------------------
# デバッグ用caput caget
debag0 = 10.0
debag1 = 10.0
debag2 = 10.0
def myCaPut( name, val ) :
    epics.caput(name,val)
    return
    #print( "caput : {} {}".format(name, val) )
    global debag0
    global debag1
    global debag2
    if name == values['name_x0'] :
        debag0 = val
    if name == values['name_x1'] :
        debag1 = val
    if name == values['name_x2'] :
        debag2 = val
    return

def myCaGet( name ) :
    return epics.caget(name)
    #print (name)
    if name == values['name_y0'] :
        val = debag0 + random.uniform(-0.01, 0.01)
    elif name == values['name_y1'] or name == 'LIiBM:SP_R0_62_2:ISNGL:KBP':
        val = debag1 + random.uniform(-0.01, 0.01)
    elif name == values['name_y2'] :
        val = debag2 + random.uniform(-0.1, 0.1)
    else :
        val = random.uniform(0, 10)
    print("caget : {} {}".format(name, val) )
    return val



#---------------------------------------------------------------
#  複数行のpythonで計算できる式をTextで渡して数値を返す
#  最後の式で評価した値を返す
#  返り値はList  [value, bool]
#  評価が成功するとvalue に数値がはいり，bool=True
#  評価が失敗すると valueにはエラー文字列が入り，bool=Falseになる．
#---------------------------------------------------------------
def calc_text(calcStr) :
    lines = calcStr.split('\n')
    #print(calcStr)
    last = ""
    for line in lines :
        print(line)
        try:
            exec(line)
        except ZeroDivisionError:
            print('division by zero')
            return [line + " <- division by zero", False]
        except NameError:
            print('undefined name')
            return [line + " <- undefined name", False]
        
        #exec(line)
        if line != "" :
            last = line

    print(last)
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
        print(ans)
        return [ans, True]



# --------------------------------------------------
#  Startが押されたときにUIから値を読むための関数
# --------------------------------------------------
def getInitSetting( values ) :
    repetition = float(values['rep'])

    dataN = int(values['dataN'])
    iterN = int(values['iterN'])
    WaitTime = float(values['WaitTime'])

    method = 'Baysian'
    if values['Baysian'] : method = 'Baysian'
    elif values['Downhill'] : method = 'Downhill'

    acquisition_weight = float(values['acquisition_weight'])
    initial_value_range = float(values['initial_value_range'])

    if dataN < 1 : dataN = 1
    if dataN > 100 : dataN = 100

    x_name_list = []
    x_min_max_list = []
    x_init_list = []
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

    limitation = values['limitation']
    functionText = values['function']
    if functionText[-1] == '\n' : functionText = functionText[:-1]

    return [repetition, dataN, iterN, WaitTime,
             x_name_list, x_min_max_list, x_init_list,
             y_name_list, y_alias_list, limitation, functionText,
             method, acquisition_weight, initial_value_range]

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
def setValueX_PV(x_name_list, newX) :
    for i in range(0, len(x_name_list) ) :
        myCaPut( x_name_list[i], newX[i] )
    return


# ----------------------------------------------
#  Y name list の Epics Recodeの値をゲットして
#  functionTextに従って値を計算しそれを返す
# ----------------------------------------------
def getValueY_PV(y_name_list, y_alias_list, functionText) :
    vals = []
    AllText = ""
    for i in range(0, len(y_name_list) ) :
        val = myCaGet(y_name_list[i]) 
        vals.append(val)
        AllText += '{}={}\n'.format(y_alias_list[i], val)
    AllText += functionText

    calcVal = calc_text(AllText)

    print(AllText),
    print("Ans=", calcVal)

    return calcVal

#--------------------------------------------------
#  min, maxで値を規格化する
#--------------------------------------------------
def normalize(x, min, max, min_nor=-1.0, max_nor=1.0) :
    #rate = (x-min)/(max-min)
    #rate_nor = rate *(max_nor-min_nor)
    #x_nor = min_nor + rate_nor
    x_nor = min_nor + ( (x-min)*(max_nor-min_nor)/(max-min) )
    return x_nor

#-------------------------------------------------
#  規格化した値を実の値に戻す
#-------------------------------------------------
def reverse_normalize(x_nor, min, max, min_nor=-1.0, max_nor=1.0) :
    x = min + (x_nor - min_nor)*(max-min)/(max_nor-min_nor)
    return x

#--------------------------------------------------
#  min, maxでlistを規格化する
#--------------------------------------------------
def normalize_list(x_list, x_min_max_list, min_nor=-1.0, max_nor=1.0) :
    x_nor_list =[]
    for x, min_max in zip(x_list, x_min_max_list) :
        x_nor_list.append( normalize(x, min_max[0], min_max[1], min_nor, max_nor) )
    return x_nor_list

#-------------------------------------------------
#  規格化したlistを実の値に戻す
#-------------------------------------------------
def reverse_normalize_list(x_nor_list, x_min_max_list, min_nor=-1.0, max_nor=1.0) :
    x_list =[]
    for x_nor, min_max in zip(x_nor_list, x_min_max_list) :
        x_list.append( reverse_normalize(x_nor, min_max[0], min_max[1], min_nor, max_nor) )
    return x_list

#--------------------------------------------------
#  min, maxで２次元listを規格化する
#--------------------------------------------------
def normalize_list2(x_list2, x_min_max_list, min_nor=-1.0, max_nor=1.0) :
    x_nor_list2 = []
    for x_list in x_list2 :
        x_nor_list2.append( normalize_list(x_list, x_min_max_list, min_nor, max_nor) )
    return x_nor_list2
        
#-------------------------------------------------
#  規格化した２次元listを実の値に戻す
#-------------------------------------------------
def reverse_normalize_list2(x_nor_list2, x_min_max_list, min_nor=-1.0, max_nor=1.0) :
    x_list2 = []
    for x_nor_list in x_nor_list2 :
        x_list2.append( reverse_normalize_list(x_nor_list, x_min_max_list, min_nor, max_nor) )
    return x_list2

# -----------------------------------------------------------
#   ベイズ最適化の1step
# -----------------------------------------------------------
def optimizationOneStep(X_values_list, Y_values_list, x_min_max_list, acquisition_weight = 2) :

    # ２次元リストの規格化
    min_nor=-1.0
    max_nor=1.0
    X_nor_values_list = normalize_list2(X_values_list, x_min_max_list, min_nor, max_nor)

    X = np.asarray( X_nor_values_list )

    Y_temp_list = []
    for i in range(0, len(Y_values_list)) :
        Y_temp_list.append([Y_values_list[i]])
    Y = np.asarray( Y_temp_list )


    bounds = [] 
    for i in range(0, len(x_min_max_list)) :
        bounds.append( {'name': 'x{}'.format(i), 'type': 'continuous',
        'domain': ( min_nor, max_nor ) } )
        # 'domain': ( x_min_max_list[i][0], x_min_max_list[i][1] ) } )

    myBopt = GPyOpt.methods.BayesianOptimization(f=None, X=X, Y=Y, domain=bounds, 
                acquisition_type='LCB',
                acquisition_weight = acquisition_weight,#LCBのパラメータを設定．デフォルトは2
                model_type='GP')
    
    x_suggest = myBopt.suggest_next_locations(ignored_X=X)
    #x_suggest = myBopt.suggest_next_locations()
    y_predict = myBopt.model.model.predict(x_suggest) #y_predictは(予測平均，予測分散)がタプルで返ってくる
    y_mean=y_predict[0]
    y_variance=y_predict[1]

    print("Next x_suggest point : ", x_suggest )
    print("y_predict mean: ", y_predict[0] ) #予測平均
    print("y_predict variance: ", y_predict[1] ) #分散

    #print(X)
    #X = np.append(X, x_suggest, axis=0)

    new_nor_x = []
    for i in range(0, len(x_min_max_list)) :
        new_nor_x.append( x_suggest[0][i] )
    
    new_x = reverse_normalize_list(new_nor_x, x_min_max_list, min_nor, max_nor)

    return new_x


#-----------------
#グラフ初期化
#-----------------
def setGraph() :
    global ax1
    global ax2

    global fig

    fig = plt.figure(figsize=(8,8))

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)

    return

#-------------------------------
# Graph描画ようY bestの遷移
#-------------------------------
def getBestValues(Y_values_list) :
    val = Y_values_list[0]
    plots = [[0], [Y_values_list[0]]]
    for i in range(0, len(Y_values_list) ) :
        if val > Y_values_list[i] :
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
    ax2.cla()

    #print(X_values_list)
    x_gp_lists = []
    for i in range(0, len(X_values_list)) :
        for j in range(0, len(X_values_list[i]) ) :
            val = normalize(X_values_list[i][j], x_min_max_list[j][0], x_min_max_list[j][1])
            if i == 0 :
                x_gp_lists.append( [ val ] )
            else :
                x_gp_lists[j].append( val )



    for i in range(0, len( x_gp_lists ) ) :
        ax1.plot(x_gp_lists[i], label="x{}".format(i))


    Y_BestValues_plot = getBestValues(Y_values_list)
    ax2.plot(Y_values_list, label='Y value')
    ax2.scatter(Y_BestValues_plot[0], Y_BestValues_plot[1], s=30, c='red', marker='o', label='best plot')
    
    ax1.set_ylabel("Normalized x")
    ax2.set_ylabel("y value")
    ax2.set_xlabel("iteration N")

    ax1.legend() #凡例を表示する場合
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
     x_name_list, x_min_max_list, x_init_list,
     y_name_list, y_alias_list, limitationText, functionText,
     method, acquisition_weight, initial_value_range] = getInitSetting( values )

    with open( fname, 'w') as f:
        print('X', file=f)
        for x_name, x_min_max, x_init in zip(x_name_list, x_min_max_list, x_init_list) :
            print('{}, {:.3f}, {:.3f}, {:.3f}'.format(x_name, x_min_max[0], x_min_max[1], x_init), file=f)
        print('', file=f)

        print('Y', file=f)

        if len(y_name_list) <= maxNY :
        #yn = 0
            #for i in range(0, maxNY) :
            for y_name, y_alias in zip(y_name_list, y_alias_list) :
                print('{}, {}'.format(y_name, y_alias), file=f)
            print('', file=f)

            print('YsettngText{', file=f)
            print('}', file=f)

        else :
            print('\nYsettngText{', file=f)
            for y_name, y_alias in zip(y_name_list, y_alias_list) :
                print('{}, {}'.format(y_name, y_alias), file=f)
            print('}', file=f)

        
        print('limitation{', file=f)
        print(limitationText, file=f)
        print('}', file=f)

        print('function{', file=f)
        print(functionText, file=f)
        print('}', file=f)

        print('repetition:{}'.format(repetition), file=f)
        print('dataN:{}'.format(dataN), file=f)
        print('iterN:{}'.format(iterN), file=f)
        print('WaitTime:{}'.format( WaitTime ), file=f)
        print('method:{}'.format( method ), file=f)
        print('acquisition_weight:{}'.format( acquisition_weight ), file=f)
        print('initial_value_range:{}'.format( initial_value_range ), file=f)

    return

#-----------------------------------------------------------
# Setting を呼び出す
#-----------------------------------------------------------
def readSetting(window, fname ) :

    xflug = False
    yflug = False
    yTextflug = False
    limitflug = False
    funcflug = False

    repetition = 1
    dataN = 1
    iterN = 1
    WaitTime = 1
    initial_value_range = 50
    x_name_list = []
    x_min_max_list = []
    x_init_list = []
    y_name_list = []
    y_alias_list = []
    limitationText = ''
    functionText = ''
    YsettingText = ''
    patternX = '(\S+),\s*(\S+),\s*(\S+),\s*(\S+)'
    patternY = '(\S+),\s*(\S+)'

    if os.path.exists(fname) :
        print(fname)
        f = open(fname, 'r')
        for line in f.readlines() :
            #print(line)

            if line == 'X\n' :
                xflug = True
            elif xflug :
                print(line)
                result = re.match(patternX, line)
                if result: 
                    x_name_list.append(result.group(1))
                    x_min_max_list.append([result.group(2), result.group(3)])
                    x_init_list.append(result.group(4))
                else :
                    xflug = False
            
            if line == 'Y\n' :
                yflug = True
            elif yflug :
                result = re.match(patternY, line)
                if result: 
                    y_name_list.append(result.group(1))
                    y_alias_list.append(result.group(2))
                else :
                    yflug = False

            if line == 'YsettngText{\n' : yTextflug = True
            elif line == '}\n' : yTextflug = False
            elif yTextflug : YsettingText += line

            if line == 'limitation{\n' : limitflug = True
            elif line == '}\n' : limitflug = False
            elif limitflug : limitationText += line

            if line == 'function{\n' : funcflug = True
            elif line == '}\n' : funcflug = False
            elif funcflug : functionText += line

            if line.find('repetition:') != -1 : repetition = line[line.find(':')+1:-1]
            if line.find('dataN:') != -1 : dataN = line[line.find(':')+1:-1]
            if line.find('iterN:') != -1 : iterN = line[line.find(':')+1:-1]
            if line.find('WaitTime:') != -1 : WaitTime = line[line.find(':')+1:-1]
            if line.find('acquisition_weight:') != -1 : acquisition_weight = line[line.find(':')+1:-1]
            if line.find('initial_value_range:') != -1 : initial_value_range = line[line.find(':')+1:-1]

            if line.find('method:Baysian') !=-1 : window['Baysian'].update(True)
            if line.find('method:Downhill') !=-1 : window['Downhill'].update(True)
            
        f.close()

        for i in range(0, maxNX ) :
            if i < len(x_name_list) :
                window['name_x{}'.format(i)].update(x_name_list[i])
                window['min_x{}'.format(i)].update(x_min_max_list[i][0])
                window['max_x{}'.format(i)].update(x_min_max_list[i][1])
                window['init_x{}'.format(i)].update(x_init_list[i])
                print(x_name_list[i])
            else :
                window['name_x{}'.format(i)].update('')
                window['min_x{}'.format(i)].update('')
                window['max_x{}'.format(i)].update('')
                window['init_x{}'.format(i)].update('')


        for i in range(0, maxNY ) :
            if i < len(y_name_list) :
                window['name_y{}'.format(i)].update(y_name_list[i])
                window['alias_y{}'.format(i)].update(y_alias_list[i])
            else :
                window['name_y{}'.format(i)].update('')
                window['alias_y{}'.format(i)].update('')

        window['YsettingText'].update(YsettingText[:-1])
        if len(YsettingText) > 5 :
            window['YTextFrame'].update(visible=True)

        window['limitation'].update(limitationText[:-1])
        window['function'].update(functionText[:-1])

        window['rep'].update(repetition)
        window['dataN'].update(dataN)
        window['iterN'].update(iterN)
        window['WaitTime'].update(WaitTime)
        window['acquisition_weight'].update(acquisition_weight)
        window['initial_value_range'].update(initial_value_range)

    return


#-------------------------------------------
# 現在の値を読み取ってInit に入れる
#-------------------------------------------
def SetCurrntValueToInit( values, window, maxNX ) :
    for i in range(0, maxNX ) :
        pvname = values['name_x{}'.format(i)]
        if len(pvname) > 0 :
            val = myCaGet(pvname)
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
            initVal = myCaGet(pvname)
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

    sg.theme('Default')

    col_Xsetting = [
        [sg.Text('            PV name', size=(31, 1)),
         sg.Text('min', size=(5, 1)), sg.Text('max', size=(5, 1)), sg.Text('init', size=(5, 1)), ], 
    ]

    maxNX = 16
    for i in range(0, maxNX) :
        col_Xsetting.append( 
            [ sg.Text( 'x{}:'.format(i), size=(2, 1)),
            sg.InputText('', size=(30, 1), key='name_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='min_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='max_x{}'.format(i) ),
            sg.InputText('', size=(6, 1), key='init_x{}'.format(i) ),
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


    maxNY = 10
    for i in range(0, maxNY) :
        col_Ysetting.append( 
            [ sg.InputText('', size=(30, 1), key='name_y{}'.format(i)),
            sg.InputText('y{}'.format(i), size=(8, 1), key='alias_y{}'.format(i) ),
            ])
    #col_Ysetting.append( [sg.Text('Evaluate function : ')] )
    #col_Ysetting.append( [sg.Multiline("", size=(40, 3), key='function') ] )

    layout1 = [
        [sg.Text("Setting file name :"), sg.InputText('', size=(80, 1), key='SettingFileNameInput'), 
         sg.Submit(button_text="Save", key='saveSettingInputText'), ],
        [sg.Submit(button_text="OpenSetting", key='openSetting'),
         sg.Text("", size=(2,1) ),
         sg.Submit(button_text="SaveSetting", key='saveSetting'), 
         sg.Text("", size=(15,1) ), sg.Text("Y setting text: ", size=(10,1) ),
         sg.Submit(button_text="ON", key='YTextOn'), sg.Submit(button_text="OFF", key='YTextOff'),], 
        [sg.Frame( ' X settings ', [[sg.Column(col_Xsetting) ]] ),
         sg.Frame( ' Y settings ', [[sg.Column(col_Ysetting) ]] ),
         sg.Frame( ' Y settings Text ', [[sg.Column(col_YsettingText) ]], visible = False , key='YTextFrame')], 
        [sg.Text("Limitation :"), sg.InputText('', size=(80, 1), key='limitation'), ],
        [sg.Text('Evaluate function : '), sg.Multiline("", size=(80, 4), key='function') ],
        [sg.Text("Beam repetition:"), sg.InputText(5, size=(7 ,1), key='rep'), sg.Text("Hz  ") , 
         sg.Text(" data N at a point:"), sg.InputText(10, size=(7 ,1), key='dataN'), 
         sg.Text(" Iteration N :"), sg.InputText(50, size=(7 ,1), key='iterN'),
         sg.Text(" Wait Time [sec]:"), sg.InputText(1, size=(7 ,1), key='WaitTime'), ], 
        [ sg.Frame( ' Optimaze method ', [
         [sg.Radio('Bayesian optimization   ',  key='Baysian', group_id='0', default=True), 
         sg.Text("acquisition_weight:"), sg.InputText(1, size=(5 ,1), key='acquisition_weight'), sg.Text("defalt:2, exploration:3") , ],
         [sg.Radio('Downhill simplex        ', key='Downhill', group_id='0'), 
         sg.Text("initial value range:"), sg.InputText(50, size=(5 ,1), key='initial_value_range'), sg.Text("%") ,], ]) ], 
        [sg.Submit(button_text="   Start   ", key='start', font=('Arial', 16) ), sg.Checkbox("with set current and shift", default=True, key = "setCurrntShift"), 
         sg.Text("", size=(2,1) ),  
         sg.Submit(button_text="  Stop  ", key='stop'),
         sg.Submit(button_text="  Restart  ", key='restart'),
         sg.Submit(button_text=" Set Best and Finish  ", key='setBestFinish'),
         sg.Text("", size=(10,1) ),
         sg.Submit(button_text="  Abort  ", key='abort', button_color=('white', 'red')), ],
        [sg.Text("stop", size=(100 ,1), key='info'),],

        [ sg.Multiline("", size=(110, 5), key='log')],
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
    window['name_x0'].update('LIiMG:PX_R0_02:IWRITE:KBP')
    window['min_x0'].update(-3)
    window['max_x0'].update(3)
    window['init_x0'].update(1)
    window['name_x1'].update('LIiMG:PY_R0_02:IWRITE:KBP')
    window['min_x1'].update(-3)
    window['max_x1'].update(3)
    window['init_x1'].update(-1)

    window['name_y0'].update('LIiBM:SP_R0_62_1:ISNGL:KBP')
    window['name_y1'].update('LIiBM:SP_R0_62_2:ISNGL:KBP')
    window['function'].update('(y0-0.5)**2+(y1-1.5)**2')

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
    Y_values_list = []

    bestY_value = 9999
    bestXset = []

    while True: #ループに入る
        event, values = window.read(timeout=t,timeout_key='-timeout-') #timeoutの単位はms
        if event is None:
            print('exit')
            break

        if event == 'openSetting':
            fname = sg.popup_get_file('open file', file_types=(("text Files", ".txt"), ("all Files", "*.*")) )
            if type(fname) is str and len(fname)>0 :
                readSetting(window, fname )
                window['SettingFileNameInput'].update(fname)
        if event == 'saveSetting':
            fname = sg.popup_get_file('save as', save_as=True, file_types=(("text Files", ".txt"), ("all Files", "*.*")) )
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

            bestY_value = 9999
            bestXset = []
            log_text += "start\n" 

            [repetition, dataN, iterN, WaitTime,
             x_name_list, x_min_max_list, x_init_list,
             y_name_list, y_alias_list, limitationText, functionText, 
             method, acquisition_weight, initial_value_range] = getInitSetting( values )

            if values["setCurrntShift"] :
                #"with set current and shift" にチェックが入っていたら現在値を読み込んで範囲もシフト
                [x_init_list, x_min_max_list] = SetCurrntValueToInit_and_ShiftMinMaxToInit( values, window, maxNX )


            #if repetition < 1 : repetition = 1
            if repetition > 50 : repetition = 50
            t = 1000/repetition #timeoutして自動更新する時間を決める

            newX = x_init_list # newX にinitを与える

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

            if method == 'Downhill' :
                downhill = dhs.Downhill_simplex()
                downhill.setLimit(len(x_min_max_list), x_min_max_list)
                initX_list_DH = downhill.makeInitialXset(initial_value_range/100)
                initX_list_DH[0] = newX[:]
                initY_list_DH = [0] *len(initX_list_DH)
                new_y_list_DH = []
                print(initX_list_DH)
                print(initY_list_DH)

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


        if event == "abort":
            data_i = 0
            iter_i = 0

            mode = 'stop'

        if event == 'graph' :
            graph_view(X_values_list, Y_values_list, x_min_max_list)

        #timeout
        if mode == 'run' :

            print('Method:{} Iteration {}/{}  meas {}/{} '.format(method, iter_i+1, iterN, data_i+1, dataN,) )
            info_text = 'Method:{}, Iteration {}/{}, meas {}/{}, best y = {} at x = {} '.format(
                method, iter_i+1, iterN, data_i+1, dataN, bestY_value, bestXset)
            log_text += info_text +'\n'
            window['info'].update(info_text)
            window['info2'].update(info_text)

            if method == 'Downhill' :
                if iter_i < len(initX_list_DH) :
                    newX = initX_list_DH[iter_i]
                

            if data_i == 0 : # set value to PV
                setValueX_PV(x_name_list, newX)
                # sleep
                time.sleep(WaitTime)

                log_text += "set x {}\n".format(newX)
                Y_temp_val = 0.0

            if limitationText != '' :
                ans = getValueY_PV(y_name_list, y_alias_list, limitationText) #getValueY_PVを流用してboolを得る
                limit_bool = ans[0]
                print(limit_bool)
                if not (type(limit_bool) is bool) :
                    limit_bool = True
                    print(limit_bool)
            else :
                limit_bool = True

            if limit_bool == True :
                ans = getValueY_PV(y_name_list, y_alias_list, functionText)
                if ans[1] == False :
                    mode = 'stop'
                    sg.PopupOK( ans[0] )
                else :
                    log_text += "get y {}\n".format(ans[0])
                    Y_temp_val += ans[0]


                window['log'].update(log_text)
                data_i += 1

            if data_i == dataN :
                data_i = 0

                Y_val = Y_temp_val/dataN
                log_text += 'new X, Y = {}, {}\n'.format(newX, Y_val)

                X_values_list.append(newX)
                Y_values_list.append(Y_val)

                if Y_val < bestY_value or iter_i == 0 :
                    bestY_value = Y_val
                    bestXset = newX
                
                if method == 'Baysian' : #------- Baysian optimaze ----------------
                    # ---  GPyOpt  ---
                    newX = optimizationOneStep(X_values_list, Y_values_list, x_min_max_list, acquisition_weight)

                elif method == 'Downhill' : #---- downhill simplex ----------------
                    if iter_i < len(initX_list_DH) : # 変数の数+1まではまずは値をもとめる
                        initY_list_DH[iter_i] = Y_val
                        if iter_i == len(initX_list_DH)-1 :# 数が揃ったらdownhillを始める
                            xy_list = [ [x, y] for x, y in zip(initX_list_DH, initY_list_DH) ]
                            downhill.setInitXY(xy_list)
                            new_x_stac_list = copy.deepcopy( downhill.downhill( [] ) )

                            newX = new_x_stac_list.pop(0) #list の最初の方から値をとってnewXにわたす

                    else :
                        new_y_list_DH.append(Y_val)

                        if len(new_x_stac_list) == 0 :
                            new_x_stac_list = copy.deepcopy( downhill.downhill( new_y_list_DH ) )
                            new_y_list_DH = []
                        
                        newX = new_x_stac_list.pop(0) #list の最初の方から値をとってnewXにわたす


                graph_view(X_values_list, Y_values_list, x_min_max_list)
                window['log'].update(log_text)
                iter_i += 1

                if iter_i == iterN :
                    mode = 'stop'

                    setValueX_PV(x_name_list, bestXset) #最後に最適値をセットする

                    log_text += 'Iteration {}/{}  meas {}/{} : best y = {} at x = {}\n'.format(
                    iter_i+1, iterN, data_i+1, dataN, bestY_value, bestXset)
                    log_text += 'Finish\n'

                    data_i = 0
                    iter_i = 0
                    window['log'].update(log_text)


            




    
