import optuna
import optuna_integration
import optuna_integration.botorch
import optunahub
import optuna.study.study

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate

import collections
from collections import OrderedDict
from epics import PV, camonitor

import logging
import time, datetime
import configparser
import statistics
import random

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.animation as animation

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

class DefaultListOrderedDict(OrderedDict):
    def __missing__(self,k):
        self[k] = []
        return self[k]

class View:
    def __init__(self):
        #--- Frame ---
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.fig.patch.set_facecolor('white')
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.97, hspace=0.3, wspace=0.3)
        

        # Tkinter Class
        self.root = Tk.Tk()
        
        self.root.wm_title("Injection Tuning")
        self.root.protocol('WM_DELETE_WINDOW', self._quit)
        

        #--- X設定 ---
        x_frame = Tk.LabelFrame(self.root, text="X settings")
        x_frame.pack(fill='x', padx=10, pady=10)

        headers = ['Check', 'PV name', 'Present Value', 'min', 'max']
        for i, header in enumerate(headers):
            Tk.Label(x_frame, text=header, font=('Arial', 12, 'bold'), width=12).grid(row=0, column=i)

        self.x_entries = []
        parameters = ["V-steering 1", "V-steering 2", "Septum angle", "RF phase"]
        for i in range(4):
            row_entries = []
            var = Tk.BooleanVar()
            check = Tk.Checkbutton(x_frame, variable=var)
            check.grid(row=i+1, column=0)
            row_entries.append(var)

            name = Tk.Label(x_frame, text=f'{parameters[i]}', font=('Arial', 20))
            name.grid(row=i+1, column=1)
            row_entries.append(name)

            present_value = Tk.Label(x_frame, text="N/A", font=('Arial', 12))
            present_value.grid(row=i+1, column=2)
            row_entries.append(present_value)

            min_spinbox = Tk.Spinbox(x_frame, from_=0, to=100, increment=1, width=10)
            min_spinbox.grid(row=i+1, column=3)
            row_entries.append(min_spinbox)

            max_spinbox = Tk.Spinbox(x_frame, from_=0, to=100, increment=1, width=10)
            max_spinbox.grid(row=i+1, column=4)
            row_entries.append(max_spinbox)

            detail_frame = Tk.Frame(x_frame)
            detail_frame.grid(row=i+1, column=5, columnspan=3, sticky='w')
            detail_frame.grid_remove()

            init_entry = Tk.Entry(detail_frame, width=10)
            init_entry.pack(side='left')
            step_entry = Tk.Entry(detail_frame, width=10)
            step_entry.pack(side='left')
            weight_entry = Tk.Entry(detail_frame, width=10)
            weight_entry.pack(side='left')

            row_entries.extend([init_entry, step_entry, weight_entry, detail_frame])
            self.x_entries.append(row_entries)

            toggle_button = Tk.Button(x_frame, text="more details", command=lambda df=detail_frame: self.toggle_detail(df))
            toggle_button.grid(row=i+1, column=8)

        controls_frame = Tk.Frame(self.root)
        controls_frame.pack(pady=10)

        # ボタンの設定
        Tk.Button(controls_frame, text="Optimization", command=self.Optimization).grid(row=1, column=0)
        Tk.Checkbutton(controls_frame, text="with set current and shift").grid(row=1, column=1)
        Tk.Button(controls_frame, text="Quit", command=self._quit).grid(row=1, column=2)
        Tk.Button(controls_frame, text="Restart").grid(row=1, column=3)
        Tk.Button(controls_frame, text="Set Best and Finish").grid(row=1, column=4)

        # グラフのキャンバス
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=20)

        #--- Initialization ---
        self.AssignToPV()

        # root
        self.root.update()
        self.root.deiconify()
        self.root.mainloop()

    def AssignToPV(self):
        print("AssignToPV started ...", end='')
        
        # --- Load configuration ---
        config_ini = configparser.ConfigParser()
        config_ini.read(sys.argv[1])

        self.ntrial    =   int(config_ini.get('RUN', 'ntrial'))
        self.ninit     =   int(config_ini.get('RUN', 'ninit'))
        self.sleepstep = float(config_ini.get('RUN', 'sleepstep'))
        self.sleepeval = float(config_ini.get('RUN', 'sleepeval'))

        self.dictx = OrderedDict()
        self.dicty = OrderedDict()
        self.dictc = OrderedDict()
        self.dictg = OrderedDict()        
        self.flagg = False

        nx = int(config_ini.get('PV', 'nx')) # Parameters
        ny = int(config_ini.get('PV', 'ny')) # Objective
        nc = int(config_ini.get('PV', 'nc')) # Constraint
        ng = int(config_ini.get('PV', 'ng')) # Gate, trigger, etc.

        # Parameters: Steering, Septum, Kicker, Phase, etc.
        self.pvx = []
        self.infox = []
        self.proximalweight = []
        for i in range(nx):
            pv = PV(config_ini.get('PV_X{0}'.format(i), 'name'))
            self.dictx[pv.pvname] = pv.get()
            self.pvx.append(pv)

            rmin = float(config_ini.get('PV_X{0}'.format(i), 'rmin'))
            rmax = float(config_ini.get('PV_X{0}'.format(i), 'rmax'))
            step = float(config_ini.get('PV_X{0}'.format(i), 'step'))
            wght = float(config_ini.get('PV_X{0}'.format(i), 'wght'))
            self.infox.append([rmin, rmax, step, wght])

        # Objective: Injection efficiency
        self.pvy = []
        for i in range(ny):
            pv = PV(config_ini.get('PV_Y{0}'.format(i), 'name'))
            self.pvy.append(pv)
            
            self.dicty[pv.pvname] = collections.deque(maxlen=10)
            camonitor(pv.pvname, callback=self.FetchY)

        # Constraint: Background signal level
        pvc = []
        for i in range(nc):
            pv = PV(config_ini.get('PV_C{0}'.format(i), 'name'))
            pvc.append(pv)

            self.dictc[pv.pvname] = collections.deque(maxlen=10)
            camonitor(pv.pvname, callback=self.FetchC)

        # Gate, trigger, etc.
        self.pvg = []
        for i in range(ng):
            pv = PV(config_ini.get('PV_G{0}'.format(i), 'name'))
            self.pvg.append(pv)

            camonitor(pv.pvname, callback=self.FetchG)

        print(" done.")

    def FetchY(self, pvname=None, value=None, timestamp=None, **kws):
        if self.flagg == True:
            self.dicty[pvname].append(value)
            self.dicty_median = statistics.median(self.dicty[pvname])

    def FetchC(self, pvname=None, value=None, timestamp=None, **kws):
        if self.flagg == True and value > 0.: # TODO
            self.dictc[pvname].append(value)
            self.dictc_median = statistics.median(self.dictc[pvname])

    def FetchG(self, pvname=None, value=None, timestamp=None, **kws):
        self.dictg[pvname] = value

        # True: gate open, False: gate close
        self.flagg = True
        for i, name in enumerate(self.dictg):
            if self.dictg[name] < 0.5:
                self.flagg = False

    #---
    # Optimization
    #---
    def Optimization(self):
        print("Optimization", end='')

        nstep = [0 for i in range(len(self.pvx))]

        self.itert = []
        self.xhist = DefaultListOrderedDict()
        self.yhist = []
        self.peakhold = 0
        self.peakholdhist = []
        self.colors = plt.get_cmap('tab10')  # デフォルトのカラー10色



        def objective(trial):
            # step size optimization
            dstep = OrderedDict()
            for i, pv in enumerate(self.pvx):
                x  = trial.suggest_float(pv.pvname, self.infox[i][0], self.infox[i][1])
                dx = x - self.dictx[pv.pvname]
                nstep[i] = int(abs(dx) / self.infox[i][2]) + 1
                dstep[pv.pvname] = dx / nstep[i]

            for istep in range(1, max(nstep) + 1):
                for i, pv in enumerate(self.pvx):
                    if istep <= nstep[i]:
                        while self.flagg == False:
                            print("wait for the beam gate opening...")
                            time.sleep(0.2)
                        pv.put(dstep[pv.pvname] * istep + self.dictx[pv.pvname])
                time.sleep(self.sleepstep)

            self.itert.append(trial.number)
    
            self.DrawPlot()
            self.root.update_idletasks()
            self.root.update()
            time.sleep(self.sleepeval)

            return self.dicty_median

        proximalweight = []
        for info in self.infox:
            proximalweight.append(info[-1])
        optuna_integration.botorch.botorch.set_proximal_weights_list(proximalweight)

        now = datetime.datetime.now()
        filename = now.strftime("%Y_%m_%d_%H_%M_%S")

        # optuna.logging.get_logger("optuna").addHandler(logging.FileHandler("{}.log".format(filename)))
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler("optuna.log"))
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

        study = optuna.create_study(
            sampler = optuna.integration.BoTorchSampler(),
            direction = "maximize",
            study_name="{}".format(filename),
            storage="sqlite:///optuna_test01.db",
        )

        for n in range(self.ninit):
            enqueue_params = OrderedDict()
            for i, pv in enumerate(self.pvx):
                center =    (self.infox[i][1] + self.infox[i][0]) * 0.5
                delta  = abs(self.infox[i][1] - self.infox[i][0]) * 0.1
                enqueue_params[pv.pvname] = random.uniform(center-delta, center+delta)
            study.enqueue_trial(enqueue_params)

        study.optimize(objective, n_trials=self.ntrial)

        print(f"- Best objective value: {study.best_value}")

        best_params = study.best_params
        for i, val in enumerate(best_params):
            # pvx[i].put(best_params[val])
            print(f"- Best {i} parameter: {best_params[val]}")

        optuna.visualization.plot_param_importances(study).show()
        plot_optimization_history(study).show()
        plot_parallel_coordinate(study).show()

    #---
    # DrawPlot
    #---
    def DrawPlot(self):
        self.ax1.clear()
        self.ax2.clear()

        for i, pv in enumerate(self.pvx):
            self.dictx[pv.pvname] = pv.get()
            self.xhist[pv.pvname].append(self.dictx[pv.pvname])

            self.ax1.plot(   self.itert,      self.xhist[pv.pvname],      color=self.colors(i % 10))
            self.ax1.scatter(self.itert[:-1], self.xhist[pv.pvname][:-1], color=self.colors(i % 10), s=50)
            self.ax1.scatter(self.itert[-1],  self.xhist[pv.pvname][-1],  color=self.colors(i % 10), s=50, marker='*')
        
        self.yhist.append(self.dicty_median) #加筆
        if self.dicty_median > self.peakhold :
            self.peakhold = self.dicty_median
        self.peakholdhist.append(self.peakhold)

        self.ax1.set_title('LER',   fontsize=16)
        self.ax1.set_xlabel('Step', fontsize=20)
        self.ax1.set_ylabel('Parametors', fontsize=20)
        self.ax1.set_xlim(0, 100)
        self.ax1.set_ylim(-20, 20)
        
        self.ax2.set_title('LER',   fontsize=16)
        self.ax2.set_xlabel('Step', fontsize=20)
        self.ax2.set_ylabel('Injection Efficiency', fontsize=20)
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(0, 200)
        
        
        self.ax2.plot(   self.itert, self.yhist, color="tab:blue")
        self.ax2.scatter(self.itert, self.yhist, s=50, c="tab:blue", marker='o')
        self.ax2.plot(   self.itert, self.peakholdhist, color="tab:orange")
        self.ax2.scatter(self.itert, self.peakholdhist, s=50, c="tab:orange", marker='o')
        
        # マージンの調整
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
        
        self.canvas.draw()
        plt.pause(0.01)

    def _quit(self):
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    v = View()
