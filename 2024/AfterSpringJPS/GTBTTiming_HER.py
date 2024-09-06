import datetime
import numpy as np
import matplotlib
import time

from _operator import index
from _ast import Index
from numpy import pv
from _curses import delay_output
from builtins import str
matplotlib.use('TkAgg')

import tkinter.font as tkFont

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import savefig

from epics import caget, caput, camonitor
from collections import OrderedDict

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

TRIGGER_INTERVAL = 0 # msec
DELAY_RANGE = 50
gtbtlist = "gtbtindex_her.dat"

class View:
    def __init__(self):
        self.root = Tk.Tk()
        self.root.wm_title("GTBT Delay Timing")

        #--- Binning ---
        nPVs = self.GetLineNnumber()
        #self.X, self.Y = np.meshgrid(np.arange(-DELAY_RANGE, DELAY_RANGE, 1), np.arange(0, nPVs, 1))
        #self.Z = 0.0*self.X + 0.0*self.Y + 1000
        #self.Z[0][0] = 15000 # dummy max
        self.X = np.linspace(-DELAY_RANGE, DELAY_RANGE, 2*DELAY_RANGE)
        self.Y = np.linspace(0, nPVs, nPVs)
        self.Z = np.zeros((len(self.X)-1, len(self.Y)-1))

        #--- Frame ---
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.fig.patch.set_facecolor('white')
        self.plot = self.fig.add_subplot(111)
        self.fig.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.97, hspace=0.3, wspace=0.3)
        self.fontStyle = tkFont.Font(family="Helvetica", size=20)

        #--- Plot ---
        self.plot.cla()
        self.plot.set_title('HER', fontsize=16)
        self.plot.set_xlabel('ADC DELAY', fontsize=14)
        self.plot.set_ylabel('BPMs',      fontsize=14)
        self.plot.set_xlim(-DELAY_RANGE, DELAY_RANGE)
        self.plot.set_ylim(0, nPVs)
        self.plotmesh = self.plot.pcolormesh(self.X, self.Y, self.Z, cmap='jet')
        self.fig.colorbar(self.plotmesh)
        
        #--- Other items ---
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
        self.quit = Tk.Button(master=self.root, text='Quit', command=self._quit)
        self.quit.pack(expand=1, fill=Tk.X, side='bottom')

        self.shot = Tk.Button(master=self.root, text='Screenshot and Save', command=self.Screenshot)
        self.shot.pack(expand=1, fill=Tk.X, side=Tk.BOTTOM)

        #--- Initialization ---
        now = datetime.datetime.now()
        self.fout_delay   = open("{0:%Y%m%d_%H%M%S.output_delay_HER.dat}".format(now),   'w')
        self.fout_meanadc = open("{0:%Y%m%d_%H%M%S.output_meanadc_HER.dat}".format(now), 'w')
        self.AssignToPV()

        #--- Timer starts ---
        self.TriggerTimer()

        self.root.mainloop()
    
    def GetLineNnumber(self):
        with open(gtbtlist, 'r') as fp:
            return len(fp.readlines())
        
    def AssignToPV(self):
        print("AssignToPV started...", end='')
        self.idelay = -(DELAY_RANGE + 5) # first 5 for cold runs

        self.refs  = OrderedDict()
        self.rawc  = OrderedDict()
        self.busy  = OrderedDict()
        self.start = OrderedDict()

        now = datetime.datetime.now()
        fref = open("{0:%Y%m%d_%H%M%S.delay_recover_HER.csh}".format(now), 'w')
        
        with open(gtbtlist) as fin:
            for s in fin:
                bpm = (s.split('\t')[1]).lstrip("M") # split by ":" and remove the front "M", then e.g. QC1LE

                # Make a backup of the original delay values
                self.refs["BMH:TBT:"+bpm+":CH1:DELAY:SET"] = caget("BMH:TBT:"+bpm+":CH1:DELAY:SET")
                self.refs["BMH:TBT:"+bpm+":CH2:DELAY:SET"] = caget("BMH:TBT:"+bpm+":CH2:DELAY:SET")
                self.refs["BMH:TBT:"+bpm+":CH3:DELAY:SET"] = caget("BMH:TBT:"+bpm+":CH3:DELAY:SET")
                self.refs["BMH:TBT:"+bpm+":CH4:DELAY:SET"] = caget("BMH:TBT:"+bpm+":CH4:DELAY:SET")
                
                fref.write("caput BMH:TBT:"+bpm+":CH1:DELAY:SET "+str(self.refs["BMH:TBT:"+bpm+":CH1:DELAY:SET"])+"\n")
                fref.write("caput BMH:TBT:"+bpm+":CH2:DELAY:SET "+str(self.refs["BMH:TBT:"+bpm+":CH2:DELAY:SET"])+"\n")
                fref.write("caput BMH:TBT:"+bpm+":CH3:DELAY:SET "+str(self.refs["BMH:TBT:"+bpm+":CH3:DELAY:SET"])+"\n")
                fref.write("caput BMH:TBT:"+bpm+":CH4:DELAY:SET "+str(self.refs["BMH:TBT:"+bpm+":CH4:DELAY:SET"])+"\n")

                # Allocate dictionaries  
                self.rawc[bpm+"_RAWC1"] = -1
                self.rawc[bpm+"_RAWC2"] = -1
                self.rawc[bpm+"_RAWC3"] = -1
                self.rawc[bpm+"_RAWC4"] = -1

                self.busy ["BMH:TBT:"+bpm+":BUSY"]  = -1
                self.start["BMH:TBT:"+bpm+":START"] = -1

                # Assign to PV
                camonitor("BMH:TBT:"+bpm+":RAWC1", callback=self.FetchRAWC)
                camonitor("BMH:TBT:"+bpm+":RAWC2", callback=self.FetchRAWC)
                camonitor("BMH:TBT:"+bpm+":RAWC3", callback=self.FetchRAWC)
                camonitor("BMH:TBT:"+bpm+":RAWC4", callback=self.FetchRAWC)

                camonitor("BMH:TBT:"+bpm+":BUSY",  callback=self.FetchBUSY)
                camonitor("BMH:TBT:"+bpm+":START", callback=self.FetchSTART)

        # Write the header line
        for i, key in enumerate(self.rawc.keys()):
            if i != len(self.rawc.keys())-1:
                self.fout_meanadc.write(key+",")
            else:
                self.fout_meanadc.write(key+"\n")

        for i, key in enumerate(self.rawc.keys()):
            if i != len(self.rawc.keys())-1:
                self.fout_delay.write(key+",")
            else:
                self.fout_delay.write(key+"\n")

        # Write the delay values in advance
        for idelay_tmp in range(-DELAY_RANGE, DELAY_RANGE, 1):
            for i, key in enumerate(self.refs.keys()):
                if i != len(self.refs.keys())-1:
                    self.fout_delay.write(str((self.refs[key] + idelay_tmp)%5120)+",")
                else:
                    self.fout_delay.write(str((self.refs[key] + idelay_tmp)%5120)+"\n")
        
        time.sleep(1)
        print(" done.")

    def FetchRAWC(self, pvname=None, value=None, timestamp=None, **kws):
        key = pvname.split(':')[2]+"_"+pvname.split(':')[3]
        self.rawc[key] = np.mean(value) # Mean

        delayshift = self.idelay + DELAY_RANGE
        if delayshift >= 0:
            self.Z[delayshift, list(self.rawc).index(key)] = self.rawc[key]

    def FetchBUSY( self, pvname=None, value=None, timestamp=None, **kws):
        self.busy[pvname] = value

    def FetchSTART(self, pvname=None, value=None, timestamp=None, **kws):
        self.start[pvname] = value

    def TriggerTimer(self):
        t0 = datetime.datetime.now()

        self.SetDelay() # change CH?:DELAY
        #time.sleep(1)
        
        print(self.idelay, end='')

        self.SetValue(  self.busy, 0) # change to "Ready(=0)"
        #self.CheckValue(self.busy, 0)
        #time.sleep(1)

        self.SetValue(  self.start, 0) # change to "STBY(=0)"
        self.CheckValue(self.start, 0)
        print(" : START(STBY) -> ", end='')
        #time.sleep(1)

        self.SetValue(  self.start, 1) # change to "Start(=1)"
        #self.CheckValue(self.start, 1)
        #print("START(Start) -> ", end='')
        #time.sleep(1)
        
        self.CheckValue(self.busy, 1) # should be "Busy(=1)"
        print("BUSY(Busy) -> ", end='')
        #time.sleep(1)

        caput("TM_TRN:CCC_1_2:SYNC", 520, wait=True)
        #time.sleep(1)

        self.CheckValue(self.busy, 0) # should be "Ready(=0)"
        print("BUSY(Ready) -> Done.")

        self.DrawPlot()

        # write mean ADC values
        delayshift = self.idelay + DELAY_RANGE
        if delayshift >= 0:
            for i, key in enumerate(self.rawc.keys()):
                if i != len(self.rawc.keys())-1:
                    self.fout_meanadc.write(str(self.Z[delayshift, list(self.rawc).index(key)])+",")
                else:
                    self.fout_meanadc.write(str(self.Z[delayshift, list(self.rawc).index(key)])+"\n")

        self.idelay += 1

        if self.idelay == DELAY_RANGE:
            time.sleep(1)
            self.Screenshot()
            sys.exit()
        
        t1    = datetime.datetime.now()
        tdiff = datetime.timedelta(microseconds = (TRIGGER_INTERVAL*1000)) - (t1 - t0)
        
        #self.root.after(int(tdiff.microseconds/1000.), self.TriggerTimer)
        self.root.after(1000, self.TriggerTimer)

    def SetDelay(self):
        for key in self.refs.keys():
            caput(key, (self.refs[key] + self.idelay)%5120, wait=True)
            
    def SetValue(self, mydict, setval):
        for key in mydict.keys():
            caput(key, setval, wait=True)
            
    def CheckValue(self, mydict, setval):
        while True:
            allgood = True # initialize
            for key, val in mydict.items():
                if val != setval:
                    allgood = False

            if allgood == True:
                break
            else:
                time.sleep(0.1)
            
    def DrawPlot(self):
        self.plotmesh.set_array((self.Z[:-1,:-1]).ravel())
        self.canvas.draw()
                
    def Screenshot(self):
        now = datetime.datetime.now()
        self.fig.savefig("{0:%Y%m%d_%H%M%S.png}".format(now), dpi=100)
        plt.close(self.fig)
    
    def _quit(self):
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    v = View()