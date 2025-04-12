import numpy as np
import motplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict

class State():
#この State クラスは、グリッド上の状態（マス目の位置）を表すためのクラスです。
    def __init__(self,row=-1,column=-1):#row:行番号,column:列番号
        self.row=row
        self.column=column

    def __repr__(self):#オブジェクトの文字列表現を返す関数
        return "<State:[{},{}]>".format(self.row,self.column)
    
    def clone(self):#現在の状態をコピーする関数
        return State(self.row,self.column)
    
    def __hash__(self):#これはハッシュ関数。State オブジェクトを辞書のキーや集合(set)の要素として使えるようにするために必要です。
        return hash((self.row,self.column))
    
    def __eq__(self,other):#等価比較（==）のための関数
        return self.row == other.row and self.column == other.column