from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QFileDialog, QMessageBox
from PyQt5 import QtWebEngineWidgets
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from maindata import *

import numpy as np

import plotly.express as px
import plotly.graph_objects as go


def open():
    try:
        window_new = MainWindow(QFileDialog().getOpenFileName()[0])
        window_new.showMaximized()
    except FileNotFoundError:
        print('Error')


class MainWindow(QMainWindow):
    def __init__(self, link='Z:\\2-UGIR\ONtrim\Нестеров С\Конференция КНТК 2024\Form.xlsx'):
        super().__init__()
        uic.loadUi(r"C:\Users\SE_Nesterov\PycharmProjects\Data\Test.ui", self)
        self.link = link
        self.boreholes = pd.ExcelFile(self.link).sheet_names

        self.borehole = Borehole(self.boreholes[0], self.link)
        self.params = self.borehole.params.copy()

        self.comboBox.addItems(self.boreholes)
        self.comboBox_2.addItems(self.params)
        self.comboBox_3.addItems(self.params)
        self.comboBox_4.addItems(self.params)

        self.pushButton.clicked.connect(lambda: self.dbscan_hand(self.doubleSpinBox.value()))
        self.pushButton_2.clicked.connect(self.plot_dist)
        self.pushButton_3.clicked.connect(self.filtration)
        self.pushButton_4.clicked.connect(self.back)
        self.pushButton_5.clicked.connect(self.delete)
        self.pushButton_6.clicked.connect(self.isolationforest)
        self.pushButton_7.clicked.connect(self.dbscan_auto)
        self.pushButton_8.clicked.connect(self.part)

        self.comboBox.textActivated.connect(self.init)
        self.comboBox_2.textActivated.connect(self.plot)
        self.comboBox_3.textActivated.connect(self.plot)
        self.comboBox_4.textActivated.connect(self.plot)

        self.actionSave.triggered.connect(self.writer)
        self.actionOpen.triggered.connect(open)

        self.checkBox.stateChanged.connect(self.plot)

        self.spinBox_2.valueChanged.connect(self.plot)

    def init(self, name):
        self.borehole = Borehole(name, self.link)

        self.date()
        self.plot()

    def date(self):
        self.dateEdit.setMinimumDate(self.borehole.df.index.tolist()[0])
        self.dateEdit.setMaximumDate(self.borehole.df.index.tolist()[-1])

        self.dateEdit_2.setMinimumDate(self.borehole.df.index.tolist()[0])
        self.dateEdit_2.setMaximumDate(self.borehole.df.index.tolist()[-1])
        self.dateEdit_2.setDate(self.borehole.df.index.tolist()[-1])

    def plot(self):
        self.fig = px.scatter(self.borehole.df, y=self.comboBox_2.currentText(), color='Cluster',
                              color_continuous_scale=[(0, "crimson"), (1, "darkblue")], title=self.borehole.name)
        self.fig.update_layout(xaxis_title="Data")
        self.fig.update_coloraxes(showscale=False)

        if self.checkBox.isChecked():
            self.fig.add_trace(go.Scatter(x=self.borehole.df.index,
                                          y=self.borehole.df[self.comboBox_2.currentText()].rolling(
                                              self.spinBox_2.value(), min_periods=1).mean(), line=dict(width=3),
                                          name=f'MA {self.spinBox_2.value()}'))

        self.widget_2.setHtml(self.fig.to_html(include_plotlyjs='cdn'))
        self.fig_1 = px.scatter(self.borehole.df, y=self.comboBox_3.currentText(), color='Cluster',
                                color_continuous_scale=[(0, "crimson"), (1, "darkblue")], title=self.borehole.name)
        self.fig_1.update_layout(xaxis_title="Data")
        self.fig_1.update_coloraxes(showscale=False)
        if self.checkBox.isChecked():
            self.fig_1.add_trace(go.Scatter(x=self.borehole.df.index,
                                            y=self.borehole.df[self.comboBox_3.currentText()].rolling(
                                                self.spinBox_2.value(),  min_periods=1).mean(),  line=dict(width=3),
                                            name=f'MA {self.spinBox_2.value()}'))

        self.widget_6.setHtml(self.fig_1.to_html(include_plotlyjs='cdn'))
        self.fig_2 = px.scatter(self.borehole.df, y=self.comboBox_4.currentText(), color='Cluster',
                                color_continuous_scale=[(0, "crimson"), (1, "darkblue")], title=self.borehole.name)

        self.fig_2.update_layout(xaxis_title="Data")
        self.fig_2.update_coloraxes(showscale=False)
        if self.checkBox.isChecked():
            self.fig_2.add_trace(go.Scatter(x=self.borehole.df.index,
                                            y=self.borehole.df[self.comboBox_4.currentText()].rolling(
                                                self.spinBox_2.value(),  min_periods=1).mean(), line=dict(width=3),
                                            name=f'MA {self.spinBox_2.value()}'))

        self.widget_7.setHtml(self.fig_2.to_html(include_plotlyjs='cdn'))

    def plot_dist(self):
        x_scaled = pd.DataFrame(
            StandardScaler().fit_transform(self.borehole.df))
        nbrs = NearestNeighbors(n_neighbors=self.spinBox.value()).fit(x_scaled)
        distances = np.sort(nbrs.kneighbors(x_scaled)[0][:, 1:].mean(axis=1))

        self.fig = px.line(y=distances, title=f'{self.spinBox.value()}-distance Graph',
                           labels={"x": "Data Points sorted by distance",
                                   "y": "Epsilon"})
        self.widget_2.setHtml(self.fig.to_html(include_plotlyjs='cdn'))

    def part(self):
        self.borehole.df = self.borehole.df.loc[
                           str(self.dateEdit.date().toPyDate()):str(self.dateEdit_2.date().toPyDate())].copy()
        self.plot()
        self.copy_df()

    def dbscan_auto(self):
        x_scaled = pd.DataFrame(
            StandardScaler().fit_transform(self.borehole.df))
        nbrs = NearestNeighbors(n_neighbors=self.spinBox.value()).fit(x_scaled)
        distances, indices = nbrs.kneighbors(x_scaled)
        distances = np.sort(distances[:, 1:].mean(axis=1))
        hist_array, bin_array = np.histogram(distances)
        hist_array_percent = hist_array / x_scaled.shape[0]
        s = 0
        for j, i in enumerate(hist_array_percent):
            s += i
            if s > 0.95:
                break
        self.dbscan_hand(bin_array[j + 1])
        plt.hist(distances, bins=10, edgecolor='black', linewidth=1.2)
        plt.xlabel('Среднее расстоние до ближайших соседей')
        plt.grid(alpha=0.5, linewidth=1, zorder=1)
        plt.axvline(x=bin_array[j + 1], color='red')
        plt.show()

    def dbscan_hand(self, eps):
        x_scaled = pd.DataFrame(
            StandardScaler().fit_transform(self.borehole.df))
        dbscan = DBSCAN(eps=eps, min_samples=self.spinBox.value()).fit(x_scaled)
        self.borehole.df['Cluster'] = dbscan.labels_
        self.borehole.df['Cluster'].where(~(self.borehole.df.Cluster > -1), other=0, inplace=True)

        self.plot()
        self.copy_df()

    def isolationforest(self):
        x_scaled = pd.DataFrame(
            StandardScaler().fit_transform(self.borehole.df))

        forest = IsolationForest(n_estimators=1000, contamination=self.doubleSpinBox_2.value(), max_features=6,
                                 random_state=42, bootstrap=True).fit_predict(x_scaled.values)
        self.borehole.df['Cluster'] = forest

        self.plot()
        self.copy_df()

    def filtration(self):
        self.borehole.df = self.borehole.df.loc[self.borehole.df['Cluster'] != -1].copy()

        self.date()
        self.plot()
        self.copy_df()

    def back(self):
        if len(self.borehole.df_list) > 1:
            self.borehole.df_list.pop()
        self.borehole.df = self.borehole.df_list[-1].copy()

        self.date()
        self.plot()

    def delete(self):
        try:
            self.borehole.df = pd.concat([self.borehole.df.loc[:str(self.dateEdit.date().toPyDate())].copy(),
                                          self.borehole.df.loc[str(self.dateEdit_2.date().toPyDate()):].copy()])
            if str(self.dateEdit.date().toPyDate()) == str(self.dateEdit_2.date().toPyDate()):
                self.borehole.df.drop(index=str(self.dateEdit.date().toPyDate()), inplace=True)
            else:
                self.borehole.df.drop(index=str(self.dateEdit.date().toPyDate()), inplace=True)
                self.borehole.df.drop(index=str(self.dateEdit_2.date().toPyDate()), inplace=True)
            self.date()
            self.plot()
            self.copy_df()
        except KeyError:
            QMessageBox.about(self, "Ошибка", "Нет такой даты")

    def copy_df(self):
        if not self.borehole.df.equals(self.borehole.df_list[-1]):
            self.borehole.df_list.append(self.borehole.df.copy())

    def writer(self):
        with pd.ExcelWriter(QFileDialog().getSaveFileName()[0], mode="a", engine="openpyxl",
                            if_sheet_exists="replace") as writer_save:
            self.borehole.df.iloc[:, :-1].to_excel(writer_save, sheet_name=self.borehole.name)


if __name__ == '__main__':
    app = QApplication([])

    window = MainWindow()
    window.showMaximized()

    app.exec()
