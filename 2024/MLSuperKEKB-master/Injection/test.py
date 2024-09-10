import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np

# グローバル変数
x_data = []
y_data = []

# データをリアルタイムで更新する関数
def update_data(i):
    x_data.append(i)
    y_data.append(np.sin(i * 0.1))  # ここではsin波をプロット
    ax.clear()  # グラフをクリア
    ax.plot(x_data, y_data, label='sin wave')
    ax.legend()
    ax.set_title('Real-time plot')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

# Tkinterのウィンドウを作成
root = tk.Tk()
root.title("Real-time Plot with Matplotlib")

# Tkinterフレームを作成
frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Matplotlibの図とキャンバスを作成
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Matplotlibのアニメーション機能を使用してプロットをリアルタイムで更新
ani = animation.FuncAnimation(fig, update_data, interval=100)

# Tkinterのメインループ
root.mainloop()