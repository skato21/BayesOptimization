# -*- coding: utf-8 -*-
# coding: utf-8

from xml.etree.ElementInclude import include
import os.path
import re
import random
import copy
import statistics
import numpy as np
import pandas as pd
import csv
import datetime
import logging
import sys
import threading
import time, datetime
import configparser
import TheSummer.Hitohudebayes as Hitohudebayes
import subprocess
import select
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from epics import PV, caget, caput, camonitor
import optuna
#from optuna.visualization import plot_contour
#from optuna.visualization import plot_edf
#from optuna.visualization import plot_intermediate_values
#from optuna.visualization import plot_optimization_history
#from optuna.visualization import plot_parallel_coordinate
#from optuna.visualization import plot_param_importances
#from optuna.visualization import plot_slice
#from bayeso_benchmarks import Hartmann6D, Colville
#from botorch.settings import validate_input_scaling


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
    #print("X",X)
    #print("Xold",Xold)
    #print("dx",dx)
    #print("nstep",nstep)
    #print("dstep",dstep)
    
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
def optimizationOneStep(study,x_name_list,y_name_list,y_alias_list,x_min_max_list,x_step_list,iter_i,WaitTime,dataN,threshold_list,functionText) :
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
    
    #制限付き最適化の正体
    #constraint1 = float(caget(lm_name_list[0])) - float(lossmonitor_list[0])
    #constraint2 = -float(caget(lm_name_list[1])) + float(lossmonitor_list[1])
    #for i in range(len(lm_name_list)):
        #constraint = float(caget(lm_name_list[i])) - float(lossmonitor_list[i])
        #constraint_list.append(constraint)
    #print(constraint_list)
    #trial.set_user_attr("constraints", constraint_list)
    #trial.set_user_attr("constraints", [constraint1,constraint2])
    
    Y_temp_val_list = []
    Y_temp_val = 0.0
    c_temp_list_list = []
    c_temp_val = 0.0
    data_i = 0
    
    while data_i < dataN :
    
        #getValueY_PVの入力がfunctionTextであることに注意。ans = [(functionの計算値), bool]という値になる
        #boolは処理自体が正常に終了しているかを表す。
        if len(threshold_list) != 0 :
            
            #ここの部分はloss_measureで監視した方が確実だが、今回は間に合わないのでループで処理する
            for i in range (len(threshold_list)):
                #しきい値が0のときは待機し続ける
                while caget(th_name_list[i]) > float(threshold_list[i]) :
                    time.sleep(3)
            
            ans = getValueY_th_PV(y_name_list, y_alias_list,th_name_list, th_alias_list, functionText)
            if ans[1] == False :
                print("calc_value error occurred.")
            else :
                Y_temp_val = ans[0]
                Y_temp_val_list.append(Y_temp_val)
                
                c_temp_list = []
                for i in range(len(lm_name_list)):
                    c_temp_val = float(caget(lm_name_list[i])) - float(lossmonitor_list[i]) #制限
                    c_temp_list.append(c_temp_val) #制限を格納するリスト(制限の数だけ詰める)
                    #print(c_temp_list,"c_temp_list")
                print('Iteration {}/{}  meas {}/{}  Y_temp_val = {}'.format(iter_i+1, iterN, data_i+1, dataN,Y_temp_val) )
                log_text += 'Iteration {}/{}  meas {}/{}  Y_temp_val = {}\n'.format(iter_i+1, iterN, data_i+1, dataN,Y_temp_val)
                time.sleep(1.1)
                c_temp_list_list.append(c_temp_list) #制限を格納するリストのリスト(n回のデータを詰める)
                #print(c_temp_list_list,"c_temp_list_list")
            data_i += 1
            
        elif len(threshold_list) == 0 :
            pass
        else : print("error happened in optimization.")

    #Y_val = statistics.mean(Y_temp_val_list)
    Y_val = statistics.median(Y_temp_val_list)
    c_temp_list_list_np = np.array(c_temp_list_list) #numpyに変換
    c_vals = np.median(c_temp_list_list_np, axis=0).tolist() #行列の列方向に中央値をとるその後ndarrayをlistに変換
    trial.set_user_attr("constraints", c_vals) #それぞれの制限の中央値を"constraints"に引き渡す
    #print(c_vals,"c_vals")
    print('X = {}, Y = {}'.format(X, Y_val))
    log_text += 'new X, Y = {}, {}\n'.format(X, Y_val)
    
    study.tell(trial, Y_val)
    
    return [X,Y_val,log_text]

#-----------------
#　ロスモニター監視
#-----------------
def lossMeasure(lm_name):
    global keep_running

    process = subprocess.Popen(['camonitor', lm_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    try:
        while keep_running.is_set():
            # 標準出力と標準エラー出力を監視
            readable, _, _ = select.select([process.stdout, process.stderr], [], [], 1)
            
            for stream in readable:
                if stream == process.stdout:
                    line = stream.readline().strip()
                    if line:
                        print(f"{lm_name}: {line}")  # 出力を表示

                if stream == process.stderr:
                    error_line = stream.readline().strip()
                    if error_line:
                        print(f"{lm_name} Error: {error_line}")

            time.sleep(0.1)
    except Exception as e:
        print(f"An error occurred in {lm_name}: {e}")
    finally:
        process.terminate()
        process.wait()

#-----------------------------------------------------------
# Setting を呼び出す
#-----------------------------------------------------------
def readSetting(args) :
    
    config_ini = configparser.ConfigParser()
    config_ini.read(args)

    xflug = False
    yflug = False
    yTextflug = False
    limitflug = False
    funcflug = False

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
    threshold_list = []
    lm_name_list = []
    lm_alias_list = []
    lossmonitor_list = []
    functionText = ''
    YsettingText = ''
    patternX = '(\S+),\s*(\S+),\s*(\S+),\s*(\S+)'
    patternY = '(\S+),\s*(\S+)'

    if os.path.exists(args) :
        print(args)
        
        nxr = int(config_ini.get("PV", "nxr"))
        ny = int(config_ini.get("PV", "ny"))
        nth = int(config_ini.get("PV", "nth"))
        nlm = int(config_ini.get("PV", "nlm"))
        repetition = int(config_ini.get("PV", "repetition"))
        dataN = int(config_ini.get("PV", "number_of_measurements"))
        iterN = int(config_ini.get("PV", "n_trials"))
        WaitTime = float(config_ini.get("PV", "evalsleep"))
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
            threshold = config_ini.get("PV_th{0}".format(i), "limitation")
            threshold_list.append(threshold)
            
        for i in range(nlm):
            pvname = config_ini.get("PV_lm{0}".format(i), "name")
            lm_name_list.append(pvname)
            alias = config_ini.get("PV_lm{0}".format(i), "alias")
            lm_alias_list.append(alias)
            lossmonitor = config_ini.get("PV_lm{0}".format(i), "limitation")
            lossmonitor_list.append(lossmonitor)
            
    return [repetition, dataN, iterN, WaitTime,
    x_name_list, x_min_max_list, x_init_list, x_step_list, x_weight_list,
    y_name_list, y_alias_list, th_name_list, th_alias_list, threshold_list,
    lm_name_list, lm_alias_list, lossmonitor_list, functionText,
    UCB, logEI, beta, randomvalue,gridvalue, bestvalue, enqueueData]

############################################
# ---------------- main ----------------
############################################
if __name__ == '__main__':
    log_text = ""
    
    data_i = 0
    iter_i = 0
    X_values_list = [] #2次元list
    Y_values_list = []
    bestY_value = 0
    bestXset = []
    log_text += "start\n" 
    
    args = sys.argv
    
    [repetition, dataN, iterN, WaitTime,
    x_name_list, x_min_max_list, x_init_list, x_step_list, x_weight_list,
    y_name_list, y_alias_list, th_name_list, th_alias_list, threshold_list,
    lm_name_list, lm_alias_list, lossmonitor_list, functionText,
    UCB, logEI, beta, randomvalue,gridvalue, bestvalue, enqueueData] = readSetting(args[1])
    print(lm_name_list)
    
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
    
    x_init_dict = {}
    for i in range (len(x_name_list)):
        x_init_dict[f"{x_name_list[i]}"] = x_init_list[i]
    
    now = datetime.datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    # --- Objective function ---
    study = optuna.create_study(
        # sampler=optuna.samplers.TPESampler(),
        # sampler=optuna.samplers.CmaEsSampler(source_trials=source_study.trials),
        # sampler=optuna.samplers.CmaEsSampler(),
        direction = "maximize",
        #direction = "minimize",
        # sampler=optuna.integration.BoTorchSampler(),
        study_name="{}".format(current_time),
        storage="sqlite:///SKEKB20240529.db",
        #制限あり
        sampler=Hitohudebayes.HitohudebayesSampler(x_name_list,x_min_max_list,x_weight_list,UCB,logEI,beta,n_startup_trials = 1, constraints_func=lambda trial: trial.user_attrs["constraints"]),
        #制限なし
        #sampler=Hitohudebayes.HitohudebayesSampler(x_name_list,x_min_max_list,x_weight_list,UCB,logEI,beta,n_startup_trials = 1,),
    )
    
    if randomvalue == True:
        study.enqueue_trial(x_init_dict)
        print(x_init_dict)
    
    elif gridvalue == True:
        study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+x_min_max_list[0][1])/2,f"{x_name_list[1]}" : (x_min_max_list[1][0]+x_min_max_list[1][1])/2})
        study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+3*x_min_max_list[0][1])/4,f"{x_name_list[1]}" : (x_min_max_list[1][0]+x_min_max_list[1][1])/2})
        study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+x_min_max_list[0][1])/2,f"{x_name_list[1]}" : (3*x_min_max_list[1][0]+x_min_max_list[1][1])/4})
        study.enqueue_trial({f"{x_name_list[0]}" : (x_min_max_list[0][0]+x_min_max_list[0][1])/2,f"{x_name_list[1]}" : (x_min_max_list[1][0]+3*x_min_max_list[1][1])/4})
        study.enqueue_trial({f"{x_name_list[0]}" : (3*x_min_max_list[0][0]+x_min_max_list[0][1])/4,f"{x_name_list[1]}" : (x_min_max_list[1][0]+x_min_max_list[1][1])/2})
    
    elif bestvalue == True:
        source_study = optuna.load_study(
        study_name = enqueueData,
        storage="sqlite:///SKEKB20240529.db"
    )
        # 最大化
        for trial in sorted(source_study.trials, key=lambda t: t.value)[90:]:
        
        # 最小化
        #for trial in sorted(source_study.trials, key=lambda t: t.value)[:10]:
            study.enqueue_trial(trial.params)
            print(trial.params)
            
        
    else : print("The acquisition function is not correctly selected.")
    
    filename = "./log_" + current_time + ".csv"
    
    '''
    #ロスモニターの値を並列で参照するコード
    keep_running = threading.Event()  #高級なbool値
    keep_running.set()  #True
    for lm_name in lm_name_list:
        thread = threading.Thread(target=lossMeasure, args=(lm_name,))
        thread.daemon = True  # デーモンスレッドとして設定、メインプロセスが落ちたらデーモンプロセスも落ちる
        thread.start()
    '''
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[18, 7])  # 1行2列のサブプロットを作成
    
    # 初期カラーバーの作成
    sc = ax2.scatter([], [], c=[], cmap="coolwarm", marker='o', s=100, vmin=0, vmax=100)
    cbar = fig.colorbar(sc, ax=ax2, label='Injection Efficiency')
    cbar.ax.yaxis.label.set_size(30)  # カラーバーのフォントサイズを設定
    
    plt.ion()  # インタラクティブモードをオン
    
    iter_i_list,best_value_list = [],[]
    
    
    while iter_i < iterN :
    
        [X_vals,Y_val,optimization_text] = optimizationOneStep(study,x_name_list,y_name_list,y_alias_list,x_min_max_list,x_step_list,iter_i,WaitTime,dataN,threshold_list,functionText)
        iter_i_list.append(iter_i)
        X_values_list.append(X_vals)
        Y_values_list.append(Y_val)
        best_value_list.append(study.best_value)
        log_text += optimization_text
        
        # プロットをクリア
        ax1.clear()
        # 線と点を描写
        ax1.plot(iter_i_list, Y_values_list, color="tab:blue")  # 線を描写
        ax1.scatter(iter_i_list, Y_values_list, color="tab:blue", s=100)  # 点を描写
        ax1.plot(iter_i_list, best_value_list, color="tab:orange")
        ax1.scatter(iter_i_list, best_value_list, color="tab:orange", s=100)
        
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('Trial', fontsize=30)
        ax1.set_ylabel('Injection Efficiency', fontsize=30)
        ax1.grid()
        
        # numpy配列に変換
        X_values_list_np = np.array(X_values_list)
        
        # 行列を転置する
        transposed_X_values_list_np = X_values_list_np.T
        
        # プロットをクリア
        ax2.clear()
        ax2.plot(transposed_X_values_list_np[0],transposed_X_values_list_np[1], marker='', linestyle='-', color='gray')
        # scatterプロットで、各点に色をつける
        sc = ax2.scatter(transposed_X_values_list_np[0],transposed_X_values_list_np[1], c=Y_values_list, cmap="coolwarm", marker='o', s=100, vmin=0, vmax=100)
        
        # 軸ラベルとフォントサイズの設定
        ax2.set_xlabel("param 1", fontsize=30)
        ax2.set_ylabel("param 2", fontsize=30)
        
        # グリッドの表示
        ax2.grid()
        
        # カラーバーの更新
        cbar.update_normal(sc)
        
        # 描写を更新
        plt.draw()
        plt.pause(0.01)  # 0.01秒待機
        
        
        
        #if Y_val < bestY_value or iter_i == 0 :
        if Y_val > bestY_value or iter_i == 0 :
            bestY_value = Y_val
            bestXset = X_vals
        
        with open(filename,"a") as f:
            writer = csv.writer(f)
            writer.writerow(
            [iter_i, Y_val, study.best_value] + X_vals + list(study.best_params.values())
            )
            fig.savefig("./plot" + current_time + ".png")
        
        info_text = f"best y = {study.best_value} at x = {list(study.best_params.values())}"
        log_text += info_text +'\n'
        
        iter_i += 1
    
    #keep_running.clear()  #False
    
    #graph_view(X_values_list, Y_values_list, x_min_max_list)
    setValueX_PV(x_name_list,x_step_list ,list(study.best_params.values()) , Xold, WaitTime,iter_i, iterN) #最後に最適値をセットする
    print(f"best y = {study.best_value} at x = {list(study.best_params.values())}")
    print('Finish')
    log_text += 'Finish\n'
    data_i = 0
    iter_i = 0
    
    plt.ioff()  #インタラクティブモードをオフ
    plt.show() 
    fig.savefig("./plot" + current_time + ".png")
