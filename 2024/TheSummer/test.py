import numpy as np
import matplotlib.pyplot as plt

# x軸の値を0から2πまで100点で生成
x = np.linspace(0, 2 * np.pi, 100)

# 正弦波と余弦波を計算
y_sin = np.sin(x)
y_cos = np.cos(x)

# グラフ描画
plt.figure(figsize=(10, 6))
plt.plot(x, y_sin, label='sin(x)', color='blue')
plt.plot(x, y_cos, label='cos(x)', color='orange')

# グラフのラベルやタイトルの設定
plt.title('Sin and Cos Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# グリッドの表示
plt.grid(True)

# グラフの表示
plt.show()