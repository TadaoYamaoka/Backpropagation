# -*- coding:utf-8 -*-
import sys
from PyQt4 import QtGui, QtCore
import numpy as np
import random

MARGIN = 32
WIDTH = 200
NN_X0 = MARGIN + WIDTH + MARGIN * 2
NN_Y0 = MARGIN + 50
NN_DIAMETER = 30
NN_SPACE = 100
OUT_X0 = NN_X0 + NN_SPACE * 2.5 + MARGIN * 2

DATA = []
TEST = []

# ニューラルネットワークのパラメータ
W1 = np.array([[1.0, 0.2], [0.1, 1.1]])
W2 = np.array([[1.0, 0.1], [0.1, 1.1]])
Z = np.array([[0, 0, 0]])
Y = np.array([[0, 0]])

eta = 0.005

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def softmax(u):
    e = np.exp(u)
    return e / np.sum(e)

# 順伝播
def forward(X):
    global W1
    global W2
    Z = sigmoid(X.dot(W1))
    Y = softmax(Z.dot(W2))
    return Y, Z

# 逆伝播
def back_propagation(X, Z, Y, T):
    global W1
    global W2
    D2 = Y - T
    W2_delta = Z.T.dot(D2)
    W2 -= eta * W2_delta

    sigmoid_dash = Z * (1 - Z)
    D1 = D2.dot(W2.T) * sigmoid_dash
    W1_delta = X.T.dot(D1)
    W1 -= eta * W1_delta

# メインウィンドウ
class MainWindow(QtGui.QWidget):
    # 初期化
    def __init__(self):
        super(MainWindow, self).__init__()

        # ボタン
        self.btnRandom = QtGui.QPushButton(self)
        self.btnRandom.setText("Random")
        self.btnRandom.clicked.connect(self.random)
        self.btnRandom.move(OUT_X0, MARGIN + WIDTH + MARGIN)

        # ウィンドウサイズ
        self.resize(850, 480)

    # Random
    def random(self):
        global TEST
        del TEST[:]
        for i in range(100):
            x1 = random.random()
            x2 = random.random()
            X = np.array([[x1, x2]])
            TEST.append(X)

        # 描画更新
        self.update()

    # マウスクリック
    def mousePressEvent(self, event):
        global Y
        global Z

        mx = event.pos().x()
        my = event.pos().y()
        # 入力欄
        if mx >= MARGIN and mx < MARGIN + WIDTH and my >= MARGIN and my < MARGIN + WIDTH:
            x1 = float(mx - MARGIN) / WIDTH
            x2 = float(WIDTH - (my - MARGIN)) / WIDTH
            # データに追加
            if event.button() == QtCore.Qt.LeftButton:
                data = [[x1, x2], [1, 0]]
            elif event.button() == QtCore.Qt.RightButton:
                data = [[x1, x2], [0, 1]]
            DATA.append(data)

            # 順伝播
            X = np.array([data[0]])
            Y, Z = forward(X)

            # 逆伝播
            T = np.array([data[1]])
            back_propagation(X, Z, Y, T)

            # 描画更新
            self.update()

    # 描画
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        # 入力欄グラフ
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
        painter.drawRect(MARGIN, MARGIN, WIDTH, WIDTH)
        painter.drawText(MARGIN + WIDTH / 2, MARGIN + WIDTH + 15, "x1")
        painter.drawText(MARGIN - 15, MARGIN + WIDTH / 2, "x2")

        # ニューラルネットワーク
        # 入力層
        painter.drawText(NN_X0 - 2, NN_Y0 - 15, "x")
        painter.drawEllipse(NN_X0 - NN_DIAMETER / 2, NN_Y0, NN_DIAMETER, NN_DIAMETER)
        painter.drawEllipse(NN_X0 - NN_DIAMETER / 2, NN_Y0 + NN_SPACE, NN_DIAMETER, NN_DIAMETER)
        # 隠れ層
        painter.drawText(NN_X0 + NN_SPACE - 2, NN_Y0 - 15, "z")
        painter.drawEllipse(NN_X0 + NN_SPACE - NN_DIAMETER / 2, NN_Y0, NN_DIAMETER, NN_DIAMETER)
        painter.drawEllipse(NN_X0 + NN_SPACE - NN_DIAMETER / 2, NN_Y0 + NN_SPACE, NN_DIAMETER, NN_DIAMETER)
        # 出力層
        painter.drawText(NN_X0 + NN_SPACE * 2 - 2, NN_Y0 - 15, "y")
        painter.drawEllipse(NN_X0 + NN_SPACE * 2 - NN_DIAMETER / 2, NN_Y0, NN_DIAMETER, NN_DIAMETER)
        painter.drawEllipse(NN_X0 + NN_SPACE * 2 - NN_DIAMETER / 2, NN_Y0 + NN_SPACE, NN_DIAMETER, NN_DIAMETER)
        # 教師データ
        painter.drawText(NN_X0 + NN_SPACE * 2.5 - 2, NN_Y0 - 15, "t")
        painter.drawRect(NN_X0 + NN_SPACE * 2.5 - NN_DIAMETER / 2, NN_Y0, NN_DIAMETER, NN_DIAMETER)
        painter.drawRect(NN_X0 + NN_SPACE * 2.5 - NN_DIAMETER / 2, NN_Y0 + NN_SPACE, NN_DIAMETER, NN_DIAMETER)

        # 結合
        # 入力層 - 隠れ層
        in_x1 = NN_X0 + NN_DIAMETER / 2
        in_y1 = NN_Y0 + NN_DIAMETER / 2
        in_y2 = NN_Y0 + NN_SPACE + NN_DIAMETER / 2
        z_in_x1 = NN_X0 + NN_SPACE - NN_DIAMETER / 2
        painter.drawLine(in_x1, in_y1, z_in_x1, in_y1)
        painter.drawLine(in_x1, in_y1, z_in_x1, in_y2)
        painter.drawLine(in_x1, in_y2, z_in_x1, in_y1)
        painter.drawLine(in_x1, in_y2, z_in_x1, in_y2)
        z_out_x1 = NN_X0 + NN_SPACE + NN_DIAMETER / 2
        y_in_x1 = NN_X0 + NN_SPACE * 2 - NN_DIAMETER / 2
        painter.drawLine(z_out_x1, in_y1, y_in_x1, in_y1)
        painter.drawLine(z_out_x1, in_y1, y_in_x1, in_y2)
        painter.drawLine(z_out_x1, in_y2, y_in_x1, in_y1)
        painter.drawLine(z_out_x1, in_y2, y_in_x1, in_y2)

        # 出力欄グラフ
        painter.drawRect(OUT_X0, MARGIN, WIDTH, WIDTH)
        painter.drawText(OUT_X0 + WIDTH / 2, MARGIN + WIDTH + 15, "x1")
        painter.drawText(OUT_X0 - 15, MARGIN + WIDTH / 2, "x2")

        # ニューラルネットワーク
        if len(DATA) > 0:
            # 最後に入力された値
            data = DATA[len(DATA) - 1]
            X = data[0]
            T = data[1]
            # 入力値
            painter.drawText(NN_X0 - NN_DIAMETER / 3, NN_Y0 + NN_DIAMETER / 1.5, "{0:.2f}".format(X[0]))
            painter.drawText(NN_X0 - NN_DIAMETER / 3, NN_Y0 + NN_SPACE + NN_DIAMETER / 1.5, "{0:.2f}".format(X[1]))
            # 教師データ
            painter.drawText(NN_X0 + NN_SPACE * 2.5 - NN_DIAMETER / 8, NN_Y0 + NN_DIAMETER / 1.5, "{0}".format(T[0]))
            painter.drawText(NN_X0 + NN_SPACE * 2.5 - NN_DIAMETER / 8, NN_Y0 + NN_SPACE + NN_DIAMETER / 1.5, "{0}".format(T[1]))

        # 隠れ層の値
        if Z[0,0] != 0 and Z[0,1] != 0:
            painter.drawText(NN_X0 + NN_SPACE  - NN_DIAMETER / 3, NN_Y0 + NN_DIAMETER / 1.5, "{0:.2f}".format(Z[0,0]))
            painter.drawText(NN_X0 + NN_SPACE - NN_DIAMETER / 3, NN_Y0 + NN_SPACE + NN_DIAMETER / 1.5, "{0:.2f}".format(Z[0,1]))

        # 出力の値
        if Y[0,0] != 0 and Y[0,1] != 0:
            painter.drawText(NN_X0 + NN_SPACE * 2  - NN_DIAMETER / 3, NN_Y0 + NN_DIAMETER / 1.5, "{0:.2f}".format(Y[0,0]))
            painter.drawText(NN_X0 + NN_SPACE * 2 - NN_DIAMETER / 3, NN_Y0 + NN_SPACE + NN_DIAMETER / 1.5, "{0:.2f}".format(Y[0,1]))

        # ニューラルネットワークのパラメータ
        painter.setPen(QtGui.QPen(QtCore.Qt.darkGreen, 1))
        # W1
        painter.drawText(NN_X0 + NN_SPACE * 0.5, NN_Y0 + NN_DIAMETER / 2, "{0:.3f}".format(W1[0,0]))
        painter.drawText(NN_X0 + NN_SPACE * 0.5, NN_Y0 + NN_DIAMETER / 2 + NN_SPACE / 4, "{0:.3f}".format(W1[1,0]))
        painter.drawText(NN_X0 + NN_SPACE * 0.5, NN_Y0 + NN_DIAMETER / 2 + NN_SPACE / 4 * 3, "{0:.3f}".format(W1[0,1]))
        painter.drawText(NN_X0 + NN_SPACE * 0.5, NN_Y0 + NN_DIAMETER / 2 + NN_SPACE, "{0:.3f}".format(W1[1,1]))
        # W2
        painter.drawText(NN_X0 + NN_SPACE * 1.5, NN_Y0 + NN_DIAMETER / 2, "{0:.3f}".format(W2[0,0]))
        painter.drawText(NN_X0 + NN_SPACE * 1.5, NN_Y0 + NN_DIAMETER / 2 + NN_SPACE / 4, "{0:.3f}".format(W2[1,0]))
        painter.drawText(NN_X0 + NN_SPACE * 1.5, NN_Y0 + NN_DIAMETER / 2 + NN_SPACE / 4 * 3, "{0:.3f}".format(W2[0,1]))
        painter.drawText(NN_X0 + NN_SPACE * 1.5, NN_Y0 + NN_DIAMETER / 2 + NN_SPACE, "{0:.3f}".format(W2[1,1]))

        # 入力データプロット
        penRed = QtGui.QPen(QtCore.Qt.red, 5)
        penBlue = QtGui.QPen(QtCore.Qt.blue, 5)
        for data in DATA:
            X = data[0]
            T = data[1]
            if T[0] == 1:
                pen = penRed
            else:
                pen = penBlue
            painter.setPen(pen)
            painter.drawPoint(MARGIN + X[0] * WIDTH, MARGIN + WIDTH - X[1] * WIDTH)
            painter.drawPoint(MARGIN + X[0] * WIDTH, MARGIN + WIDTH - X[1] * WIDTH)

        # テストデータのプロット
        for XT in TEST:
            YT, ZT = forward(XT)
            if YT[0,0] > 0.5:
                pen = penRed
            else:
                pen = penBlue
            painter.setPen(pen)
            painter.drawPoint(OUT_X0 + XT[0,0] * WIDTH, MARGIN + WIDTH - XT[0,1] * WIDTH)
            painter.drawPoint(OUT_X0 + XT[0,0] * WIDTH, MARGIN + WIDTH - XT[0,1] * WIDTH)

        painter.end()

    # 閉じる
    def closeEvent(self, event):
        QtCore.QCoreApplication.quit()
        sys.exit()

# main
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mainwnd = MainWindow()
    mainwnd.show()
    sys.exit(app.exec_())
