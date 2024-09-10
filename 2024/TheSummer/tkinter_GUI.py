import tkinter as tk
from tkinter import ttk
import epics
import threading
import math

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Injection tuning UI")

        # スレッドを停止するためのイベント
        self.stop_event = threading.Event()

        # タブの設定
        tab_control = ttk.Notebook(root)
        self.tab1 = ttk.Frame(tab_control)
        self.tab2 = ttk.Frame(tab_control)
        
        tab_control.add(self.tab1, text='Parameter Settings')
        tab_control.add(self.tab2, text='Other Settings')
        tab_control.pack(expand=1, fill='both')
        
        # タブ1 (X設定) のレイアウト
        self.layoutX(self.tab1)
        
        # タブ2 (その他の設定) のレイアウト
        self.layoutOther(self.tab2)
        
        # ウィンドウを閉じるイベントにスレッドを停止する処理をバインド
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def layoutX(self, frame):
        # X設定
        x_frame = tk.LabelFrame(frame, text="X settings")
        x_frame.pack(fill='x', padx=10, pady=10)
        
        headers = ['Check', 'PV name', 'Present Value', 'min', 'max']
        for i, header in enumerate(headers):
            tk.Label(x_frame, text=header, font=('Arial', 12, 'bold'), width=12).grid(row=0, column=i)
        
        self.x_entries = []
        parameters = ["V-steering 1","V-steering 2","Septum angle","RF phase"]
        for i in range(4):
            row_entries = []
            var = tk.BooleanVar()
            check = tk.Checkbutton(x_frame, variable=var)
            check.grid(row=i+1, column=0)
            row_entries.append(var)

            name = tk.Label(x_frame, text=f'{parameters[i]}', font=('Arial', 20))
            name.grid(row=i+1, column=1)
            row_entries.append(name)

            present_value = tk.Label(x_frame, text="N/A", font=('Arial', 12))
            present_value.grid(row=i+1, column=2)
            row_entries.append(present_value)

            # minとmaxのためのSpinboxウィジェット
            min_spinbox = tk.Spinbox(x_frame, from_=0, to=100, increment=1, width=10)
            min_spinbox.grid(row=i+1, column=3)
            row_entries.append(min_spinbox)

            max_spinbox = tk.Spinbox(x_frame, from_=0, to=100, increment=1, width=10)
            max_spinbox.grid(row=i+1, column=4)
            row_entries.append(max_spinbox)
            
            # 隠しフィールド
            detail_frame = tk.Frame(x_frame)
            detail_frame.grid(row=i+1, column=5, columnspan=3, sticky='w')
            detail_frame.grid_remove()  # 初期状態では隠す
            
            init_entry = tk.Entry(detail_frame, width=10)
            init_entry.pack(side='left')
            step_entry = tk.Entry(detail_frame, width=10)
            step_entry.pack(side='left')
            weight_entry = tk.Entry(detail_frame, width=10)
            weight_entry.pack(side='left')
            
            row_entries.extend([init_entry, step_entry, weight_entry, detail_frame])
            self.x_entries.append(row_entries)
            
            # 詳細表示を切り替えるボタン
            toggle_button = tk.Button(x_frame, text="more details", command=lambda df=detail_frame: self.toggle_detail(df))
            toggle_button.grid(row=i+1, column=8)
        
        controls_frame = tk.Frame(frame)
        controls_frame.pack(pady=10)
        
        tk.Button(controls_frame, text="Start").grid(row=1, column=0)
        tk.Checkbutton(controls_frame, text="with set current and shift").grid(row=1, column=1)
        tk.Button(controls_frame, text="Stop").grid(row=1, column=2)
        tk.Button(controls_frame, text="Restart").grid(row=1, column=3)
        tk.Button(controls_frame, text="Set Best and Finish").grid(row=1, column=4)
        
        # グラフのためのキャンバス
        self.canvas = tk.Canvas(frame, bg="white")
        self.canvas.pack(fill='both', expand=True, pady=20)
        
        
        
        # 臨時的なグラフの説明ラベル 
        self.info_label2 = tk.Label(frame, text="This graph is temporary", width=100, font=('Arial', 12))
        self.info_label2.pack()

        # キャンバスに正弦波を描画
        self.canvas.bind("<Configure>", self.draw_function_on_canvas)

        # PVを更新するスレッドを開始
        self.update_pvs()

    def draw_function_on_canvas(self, event=None):
        # キャンバスのサイズを取得
        self.canvas.delete("all")  # 既存の描画をクリア
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        center_y = height // 2
        amplitude = height // 4
        frequency = 2  # キャンバスの幅におけるサイクルの数
        
        points = []
        for x in range(width):
            y = center_y + amplitude * math.sin(2 * math.pi * frequency * x / width)
            points.append((x, y))
        
        # 正弦波を描く (点を結ぶ線として描画)
        for i in range(len(points) - 1):
            self.canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill="blue")

    def toggle_detail(self, frame):
        # 詳細表示の切り替え
        if frame.winfo_ismapped():
            frame.grid_remove()
        else:
            frame.grid()

    def update_pvs(self):
        # PVを定期的に更新する関数
        def read_pv():
            if self.stop_event.is_set():
                return
            for row in self.x_entries:
                pv_name = row[1].cget("text")  # ラベルのテキストからPV名を取得
                if pv_name:
                    try:
                        #value = epics.caget(pv_name)
                        row[2].config(text=str(value))
                    except Exception as e:
                        row[2].config(text="Error")
            self.root.after(1000, read_pv)
        
        self.root.after(1000, read_pv)
    
    def on_closing(self):
        # ウィンドウを閉じる際にスレッドを停止する
        self.stop_event.set()
        self.root.destroy()  # 即座にウィンドウを閉じる

    def layoutOther(self, frame):
        # 設定ファイルのコントロール
        setting_frame = tk.Frame(frame)
        setting_frame.pack(pady=10)
        
        tk.Label(setting_frame, text="Setting file name:").grid(row=0, column=0)
        self.setting_file_input = tk.Entry(setting_frame, width=80)
        self.setting_file_input.grid(row=0, column=1)
        tk.Button(setting_frame, text="Save").grid(row=0, column=2)
        tk.Button(setting_frame, text="OpenSetting").grid(row=1, column=0)
        tk.Button(setting_frame, text="SaveSetting").grid(row=1, column=2)
        tk.Label(setting_frame, text="Y setting text: ").grid(row=1, column=4)
        tk.Button(setting_frame, text="ON").grid(row=1, column=5)
        tk.Button(setting_frame, text="OFF").grid(row=1, column=6)
        
        # Y設定
        y_frame = tk.LabelFrame(frame, text="Y settings")
        y_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(y_frame, text='PV name').grid(row=0, column=0)
        tk.Label(y_frame, text='alias').grid(row=0, column=1)
        
        self.y_name = tk.Entry(y_frame, width=30)
        self.y_name.grid(row=1, column=0)
        self.y_alias = tk.Entry(y_frame, width=10)
        self.y_alias.grid(row=1, column=1)
        
        # Y設定テキスト
        y_text_frame = tk.LabelFrame(frame, text="Y settings Text")
        y_text_frame.pack(fill='x', padx=10, pady=10)
        
        self.y_text = tk.Text(y_text_frame, height=8, width=60)
        self.y_text.pack()
        
        # 評価関数
        func_frame = tk.LabelFrame(frame, text="Evaluate function")
        func_frame.pack(fill='x', padx=10, pady=10)
        
        self.func_text = tk.Text(func_frame, height=2, width=60)
        self.func_text.pack()
        
        # しきい値設定
        th_frame = tk.LabelFrame(frame, text="Threshold settings")
        th_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(th_frame, text='PV name').grid(row=0, column=0)
        tk.Label(th_frame, text='alias').grid(row=0, column=1)
        tk.Label(th_frame, text='limitation:').grid(row=0, column=2)
        
        self.th_name = tk.Entry(th_frame, width=30)
        self.th_name.grid(row=1, column=0)
        self.th_alias = tk.Entry(th_frame, width=10)
        self.th_alias.grid(row=1, column=1)
        self.th_limit = tk.Entry(th_frame, width=65)
        self.th_limit.grid(row=1, column=2)
        
        # ロスモニタ設定
        lm_frame = tk.LabelFrame(frame, text="Loss monitor settings")
        lm_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(lm_frame, text='PV name').grid(row=0, column=0)
        tk.Label(lm_frame, text='alias').grid(row=0, column=1)
        tk.Label(lm_frame, text='limitation:').grid(row=0, column=2)
        
        self.lm_entries = []
        for i in range(3):
            lm_row = []
            name = tk.Entry(lm_frame, width=30)
            name.grid(row=i+1, column=0)
            alias = tk.Entry(lm_frame, width=10)
            alias.grid(row=i+1, column=1)
            limit = tk.Entry(lm_frame, width=65)
            limit.grid(row=i+1, column=2)
            lm_row.extend([name, alias, limit])
            self.lm_entries.append(lm_row)
        
        # 下部のコントロール
        controls_frame = tk.Frame(frame)
        controls_frame.pack(pady=10)
        
        tk.Label(controls_frame, text="Beam repetition:").grid(row=0, column=0)
        self.rep_input = tk.Entry(controls_frame, width=7)
        self.rep_input.grid(row=0, column=1)
        
        tk.Label(controls_frame, text="data N at a point:").grid(row=0, column=2)
        self.dataN_input = tk.Entry(controls_frame, width=7)
        self.dataN_input.grid(row=0, column=3)
        
        tk.Label(controls_frame, text="Iteration N:").grid(row=0, column=4)
        self.iterN_input = tk.Entry(controls_frame, width=7)
        self.iterN_input.grid(row=0, column=5)
        
        tk.Label(controls_frame, text="Wait Time [sec]:").grid(row=0, column=6)
        self.wait_time_input = tk.Entry(controls_frame, width=7)
        self.wait_time_input.grid(row=0, column=7)
        
        
        
        self.info_label = tk.Label(controls_frame, text="stop", width=100)
        self.info_label.grid(row=2, column=0, columnspan=8)
        
        # ログ
        self.log_text = tk.Text(frame, height=8, width=120)
        self.log_text.pack(pady=10)

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
