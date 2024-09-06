# -*- coding: utf-8 -*-
# coding: utf-8

# 23/03/22 limitから外れたときに評価値がマイナスだとうまく動かないバグを直した
# 23/03/20 limit_listのmin, maxの順番が逆でも自動で治すようにした．
# 23/03/07
# 関数を渡さない形のdownhillを作るた
# classによって実装するのが現実的だと考えた
#

# 最小化関数は次回の評価点を返すしながら，次回にその評価値を受け取るという流れで最小化を行う．
# その最小化関数プロセスは，いくつかのphaseに分けられ，評価値待ちのときのphaseをクラスの中で記憶しておく．
# 次回評価値を渡されたときに，どこのphaseの待ちだったかを思い出して再出発するようにする．
# 待ちが発生するのは，
# 1. 反射点評価待ち
# 2. 膨張点評価待ち
# 3. １次元収縮点評価待ち
# 4. 全体収縮評価点群の評価待ち
# である．


import copy
import random

import numpy as np
import matplotlib.pyplot as plt

def npAve( listVec ) :
    #print(len(listVec))
    sum = np.array( [0.0]*len(listVec) )
    for vec in listVec :
        #print(vec)
        sum += vec
    ave = sum/len(listVec)
    return ave

###################################################
# 評価点が指定範囲内にいるかどうかを判断する
# 範囲内ならTrueを返す．範囲ならFalse
# ただし，データ数がおかしくてもFalseを返す
###################################################
def isIn_limit(xa, limit_list) :
    if len(xa) == len(limit_list) :
        for i in range(0, len(xa) ) :
            if xa[i] < limit_list[i][0] :
                return False
            if xa[i] > limit_list[i][1] :
                return False
        return True
    else :
        return False


class Downhill_simplex :
    """Downhill Simplex class. Don't use ovject function """

    def __init__(self):
        self.phase = 0
        self.limit_list = []
        self.xy_list = []

    def clear(self) :
        self.phase = 0
        self.limit_list = []
        self.xy_list = []

    def phaseIs(self) :
        return self.phase

    def maxNormOfX(self) :
        if len(self.xy_list) >= 2 :
            x0 = self.xy_list[0][0]
            xN = self.xy_list[self.variableN][0]
            dist = np.linalg.norm( np.array(x0) - np.array(xN) )
            return dist
        else :
            return 0


    def setLimit(self, n, limit_list ) :
        if len(limit_list) != n : return False

        self.variableN = n
        self.limit_list = copy.deepcopy(limit_list)

        for minmax in self.limit_list :# 23/03/20 limit_listのmin, maxの順番が逆でも自動で治すようにした．
            min = minmax[0]
            max = minmax[1]
            if max < min :
                minmax[1] = min
                minmax[0] = max

        return True


    def setInitXY(self, xy_list) :
        if len(xy_list)-1 != self.variableN : return False
        self.xy_list = copy.deepcopy(xy_list)

        return True


    def makeInitialXset(self, range_ratio = 0.5) :
        n = self.variableN
        initX_list = []
        for i in range(0, n+1) :
            randList = []
            for j in range(0, n) :
                abs_range = (self.limit_list[j][1] -self.limit_list[j][0])*range_ratio
                center = (self.limit_list[j][1] +self.limit_list[j][0])/2
                min = center - abs_range/2
                max = center + abs_range/2
                randList.append( random.uniform( min, max ) )
            initX_list.append(np.array(randList))

        return initX_list


    def downhill(self, new_y_list = [] ) : #スパゲッティ注意!!!!!!!!!!!!
        n=self.variableN
        while True :
            if self.phase == 0 :
                self.xy_list = sorted(self.xy_list , key = lambda xy : xy[1] )
                #反射
                #最悪点（最大点）をその他の点の重心の反対側へ持っていく
                self.xn0 = npAve( [ xy[0]  for xy in self.xy_list[0: n] ]  ) #x_list[n]が最悪点
                #print(self.xn0)
                #print(xy_list[n][0])
                self.new_x = self.xn0 +( self.xn0 - self.xy_list[n][0] )

                self.phase = 1 # 反射点待ち
                if isIn_limit(self.new_x, self.limit_list) : return [self.new_x]
                else : new_y_list = [ self.xy_list[n][1]+1 ] # 範囲外に出ていたら擬似的に最悪点より悪い値を与えて次に進む

            # 反射点待ち
            if self.phase == 1 :
                self.new_y = new_y_list[0]

                # 膨張 ?
                if self.new_y <= self.xy_list[0][1] : #最良点よりも新たな点が良かったか
                    #膨張 : 更に２倍進んで見る
                    self.new_x2 = self.xn0 +2*( self.xn0 -self.xy_list[n][0] )
                    self.phase = 2 # 膨張待ち
                    if isIn_limit(self.new_x2, self.limit_list) : return [self.new_x2]
                    else : new_y_list = [ self.xy_list[n][1]+1 ] # 範囲外に出ていたら擬似的に最悪点より悪い値を与えて次に進む

                elif self.new_y < self.xy_list[n-1][1] : #少なくとも現在の2番目の最悪点よりはいいときにその点と置き換え
                    #反射した点へ置き換え
                    self.xy_list[n] = [self.new_x, self.new_y]
                    self.phase = 0 #最初にもどる

                else :  # 現在の最悪点より悪い場合
                    #一次元収縮
                    self.new_x = self.xn0 -( self.xn0 -self.xy_list[n][0] )/2
                    self.phase = 3 #一次元収縮待ち
                    return [self.new_x]


            # 膨張待ち
            if self.phase == 2 :
                self.new_y2 = new_y_list[0]

                if self.new_y < self.new_y2 : 
                    #反射した点へ置き換え
                    self.xy_list[n] = [self.new_x, self.new_y]
                else :
                    #反射and膨張した点へ置き換え
                    self.xy_list[n] = [self.new_x2, self.new_y2]

                self.phase = 0 #最初にもどる

            #一次元収縮待ち
            if self.phase == 3 :
                self.new_y = new_y_list[0]
                if self.new_y <= self.xy_list[n][1] : #１次元収縮の結果が現在の最悪点より良ければ
                #if self.new_y <= self.xy_list[1][1] : # Test for 全体収縮
                    #置き換え
                    self.xy_list[n] = [self.new_x, self.new_y]
                    self.phase = 0 #最初にもどる

                else : #全体収縮
                    self.new_x_list = []
                    for i in range( 1, len(self.xy_list) ) :
                        self.new_x_list.append( self.xy_list[0][0] + (self.xy_list[i][0]-self.xy_list[0][0])/2 )

                    self.phase = 4 # 全体収縮待ち
                    #print("全体収縮", self.new_x_list)
                    return self.new_x_list

            #全体収縮待ち
            if self.phase == 4 :
                for i in range( 1, len(self.xy_list) ) :
                    self.xy_list[i][0] = self.new_x_list[i-1]
                    self.xy_list[i][1] = new_y_list[i-1]

                self.phase = 0 #最初にもどる





def testFunc(x, args=[]) :
    return 6*(x[0]-6.0)**2 + 2*(x[1]-7.0)**2 


if __name__ == '__main__':
    # --------可視化グラフのためのマップ作り----------
    mapx = np.array([ float(i) for i in range(0, 11)])
    mapy = np.array([ float(i) for i in range(0, 11)])
    mapxx, mapyy = np.meshgrid(mapx, mapy)
    mapval = []
    for y in mapy :
        tempList = []
        for x in mapx :
            tempList.append( testFunc( [x, y] ) )
        mapval.append( np.array(tempList) )
    #plt.contourf(mapxx, mapyy, mapval)
    #plt.show()
    # ---------------------------------------------


    n=2
    #initX_list = [ np.array([0.5, 3]), np.array([3, 2.]), np.array([2., 0.5]), ]
    #initX_list = [ np.array([1.5, 3]), np.array([3, 2.]), np.array([7, 7]), ]
    #initX_list = [  ]

    limit_list = [[0, 10], [0, 10]]
    

    dhs = Downhill_simplex()
    dhs.setLimit(n, limit_list)
    initX_list = dhs.makeInitialXset(0.2)

    xy_list = [ [x, testFunc(x) ] for x in initX_list ]
    print(xy_list)
    dhs.setInitXY(xy_list)

    # 一回のループごとにグラフを描きながらDownHill
    new_y_list = []
    for i in range(0, 30) :
        new_x_list = dhs.downhill( new_y_list )
        new_y_list = [ testFunc(x) for x in new_x_list ]
        #print(new_x_list) 
        #print(new_y_list) 
        #if len(new_x_list) > 1 : print(new_y_list) 
        

        gxl = [ x[0][0] for x in dhs.xy_list]
        gxl.append(dhs.xy_list[0][0][0])
        gyl = [ x[0][1] for x in dhs.xy_list]
        gyl.append(dhs.xy_list[0][0][1])

        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.contourf(mapxx, mapyy, mapval, cmap='hsv', levels=40)
        #plt.pcolormesh(mapxx, mapyy, mapval, cmap='hsv')
        plt.plot(gxl, gyl, color='black')
        x, y = zip(*new_x_list)
        plt.plot(x, y, 'bo')
        plt.grid() #グリッドを入れる
        plt.savefig('graph_test/{:0>3d}.png'.format(i) )
        plt.cla()
        
