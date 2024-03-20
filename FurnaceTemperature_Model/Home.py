import sys
import csv
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtWidgets, uic,QtCore
from PyQt5.QtGui import QPixmap, QPalette,QBrush,QMovie,QShowEvent
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import stats
import random
from Prediction_Models import PredLinearRegressionModel,PredDecisionTreeModel,PredDeeplearningModel,PredRandomForestModel


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('Main.ui', self)
        self.setWindowTitle("Home screen")
        # self.show()

        self.button = self.findChild(QtWidgets.QPushButton, 'bt_selectFile')
        self.button1 = self.findChild(QtWidgets.QPushButton, 'pTest')
        self.button1.clicked.connect(self.button_pTest)

        self.CleandataBtn = self.findChild(QtWidgets.QPushButton, 'bt_cleandata' )
        self.CleandataBtn.clicked.connect(self.CleandataFun)
        
        self.OpenPredPage = self.findChild(QtWidgets.QPushButton, 'bt_PredPage')
        self.OpenPredPage.clicked.connect(self.OpenPage2)

        self.OpenPredPage = self.findChild(QtWidgets.QPushButton, 'bt_AnalysisPage')
        self.OpenPredPage.clicked.connect(self.OpenPage3)


        self.container = self.findChild(QtWidgets.QWidget, 'Statwidget')
        self.figure = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvas = FigureCanvas(self.figure)  
        layout = QVBoxLayout(self.container)  
        layout.addWidget(self.canvas)

        self.container = self.findChild(QtWidgets.QWidget, 'Chart_Scatter')
        self.figureScatter = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasScatter = FigureCanvas(self.figureScatter) 
        layout = QVBoxLayout(self.container)  
        layout.addWidget(self.canvasScatter)

        self.containerAfterfilter = self.findChild(QtWidgets.QWidget, 'Chart_Scatter_Afterfilter')
        self.figureScatterAfterfilter = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasScatterAfterfilter = FigureCanvas(self.figureScatterAfterfilter)  
        layout = QVBoxLayout(self.containerAfterfilter)  
        layout.addWidget(self.canvasScatterAfterfilter)

        self.containerStatwidgetAfterFilter = self.findChild(QtWidgets.QWidget, 'StatwidgetAfterFilter')
        self.figureStatwidgetAfterFilter = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasfigureStatwidgetAfterFilter = FigureCanvas(self.figureStatwidgetAfterFilter)  
        layout = QVBoxLayout(self.containerStatwidgetAfterFilter)  
        layout.addWidget(self.canvasfigureStatwidgetAfterFilter)



   

        self.StatusLabel = self.findChild(QtWidgets.QLabel, 'Label_status')
        self.show()

      
        
        # self.loaderLabel = self.findChild(QtWidgets.QLabel, 'label_4')
        # self.movie = QMovie("ABC.gif")
        # self.loaderLabel.setMovie(self.movie)
        # self.startAnimation()

    def showEvent(self, event: QShowEvent):
           
        self.csv_file_path="D:\Projects\TCE\FurnaceTemperature_Model\FurnaceData.csv"
        self.load_csv_data(self.csv_file_path)
        self.button_pTest()

    
    def select_csv(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("CSV Files (*.csv)")
        file_dialog.exec_()

        selected_files = file_dialog.selectedFiles()
        if selected_files:
            self.csv_file_path = selected_files[0]
            self.load_csv_data(self.csv_file_path)

     #stat       
    def button_pTest(self):
        text_edit = self.findChild(QtWidgets.QTextEdit, 'textEdit_describe')
        #data = pd.read_csv(self.csv_file_path)
        data = pd.read_csv(self.csv_file_path, usecols=[1, 2, 3,4])
        statistics=data.describe()
        text_edit.setPlainText(str(statistics))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        statistics.plot(kind='bar', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.subplots_adjust(bottom=0.2)

        # df = pd.read_csv(self.csv_file_path)
        # df['z-score'] = (df['Value1'] - df['Value1'].mean()) / df['Value1'].std()
        ColumnwithVal = ['PreHeatZone','HeatZone1','HeatZone2','SockingZone']
        # Filtereddf = data[ColumnwithVal]
        # z_scores = (Filtereddf - Filtereddf.mean()) / Filtereddf.std()

        
       #plot Scatter
        self.figureScatter.clear()
        ax = self.figureScatter.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        for column in ColumnwithVal:
           z_scores1 = stats.zscore(data[column])
           color = ['red' if value <= -1 or value >= 1 else 'green' for value in z_scores1]
           ax.scatter(z_scores1.index, z_scores1, marker='o', color=color)  
        plt.subplots_adjust(bottom=0.1)
        self.canvas.draw()
        self.canvasScatter.draw()


            

    

    
    def load_csv_data(self, file_path):
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(0)  # Clear existing columns

        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            csv_data = list(csv_reader)
            if len(csv_data) > 0:
                num_rows = len(csv_data)
                num_columns = len(csv_data[0])
                self.tableWidget.setRowCount(num_rows)
                self.tableWidget.setColumnCount(num_columns)

                for row in range(num_rows):
                    for column in range(num_columns):
                        item = QTableWidgetItem(csv_data[row][column])
                        self.tableWidget.setItem(row, column, item)



    def read_csv_file(self, file_path):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                print(row)
            # Do something with the selected CSV file path
    def CleandataFun(self):
        data = pd.read_csv(self.csv_file_path)
        ColumnwithVal = ['PreHeatZone','HeatZone1','HeatZone2','SockingZone']
        #print(data)
        for column in ColumnwithVal:
            Colmedian = data[column].median()
            abs_deviation = abs(data[column]-Colmedian)
            print(abs_deviation)
            Percentage_deviation = abs_deviation / Colmedian * 100
            data.loc[Percentage_deviation > 40,column] = Colmedian
        
        #plot Scatter
        self.figureScatterAfterfilter.clear()
        ax = self.figureScatterAfterfilter.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        for column in ColumnwithVal:
           z_scores1 = stats.zscore(data[column])
           color = ['red' if value <= -1 or value >= 1 else 'green' for value in z_scores1]
           ax.scatter(z_scores1.index, z_scores1, marker='o', color=color)  
        plt.subplots_adjust(bottom=0.1)
        self.canvasScatterAfterfilter.draw()

        self.CleanedData = data

        #Plot Bar chart
        #canvasfigureStatwidgetAfterFilter

   

        #Plot Bar chart
        text_edit = self.findChild(QtWidgets.QTextEdit, 'textEdit_describe')
        statistics=data.iloc[:, 1:5].describe()
        text_edit.setPlainText(str(statistics))
        self.figureStatwidgetAfterFilter.clear()
        ax = self.figureStatwidgetAfterFilter.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        statistics.plot(kind='bar', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.subplots_adjust(bottom=0.2)
        self.canvasfigureStatwidgetAfterFilter.draw()

    def OpenPage2(self):
        self.second_page = SecondPage()
        self.setCentralWidget(self.second_page) 
    def OpenPage3(self):
        self.Third_page = ThirdPage()
        self.setCentralWidget(self.Third_page) 

class SecondPage(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('Page2.ui', self)
        self.buttonSelectCSV = self.findChild(QtWidgets.QPushButton, 'bt_selectFile')
        self.buttonSelectCSV.clicked.connect(self.select_csv)
        self.buttonPredict = self.findChild(QtWidgets.QPushButton, 'bt_predict')
        self.buttonPredict.clicked.connect(self.Predict)

        self.button1PredCokeoven = self.findChild(QtWidgets.QPushButton, 'bt_predict_Cokeoven')
        self.button1PredCokeoven.clicked.connect(self.PredictCokeoven)

        self.button2Mainpage = self.findChild(QtWidgets.QPushButton, 'bt_Mainpage')
        self.button2Mainpage.clicked.connect(self.GoMainpage)


        self.containerDataselected = self.findChild(QtWidgets.QWidget, 'Wid_Dataselected')
        self.figureDataselected = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasDataselected = FigureCanvas(self.figureDataselected)  
        layoutDataselected = QVBoxLayout(self.containerDataselected)  
        layoutDataselected.addWidget(self.canvasDataselected)

        self.containerRateofchange = self.findChild(QtWidgets.QWidget, 'Wid_Rateofchange')
        self.figureRateofchange = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasRateofchange = FigureCanvas(self.figureRateofchange)  
        layoutRateofchange = QVBoxLayout(self.containerRateofchange)  
        layoutRateofchange.addWidget(self.canvasRateofchange)

        self.containerPredictionChart = self.findChild(QtWidgets.QWidget, 'Wid_PredictionChart')
        self.figurePredictionChart = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasPredictionChart = FigureCanvas(self.figurePredictionChart)  
        layoutPredictionChart = QVBoxLayout(self.containerPredictionChart)  
        layoutPredictionChart.addWidget(self.canvasPredictionChart)

        self.containerPredictionChart2 = self.findChild(QtWidgets.QWidget, 'Wid_PredictionChart2')
        self.figurePredictionChart2 = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasPredictionChart2 = FigureCanvas(self.figurePredictionChart2)  
        layoutPredictionChart2 = QVBoxLayout(self.containerPredictionChart2)  
        layoutPredictionChart2.addWidget(self.canvasPredictionChart2)

    def GoMainpage(self):
        self.Mainpage = MyApp()
        self.setCentralWidget(self.Mainpage)

        
    def select_csv(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("CSV Files (*.csv)")
        file_dialog.exec_()

        selected_files = file_dialog.selectedFiles()
        if selected_files:
            self.csv_file_path = selected_files[0]
            print(self.csv_file_path)
            df = pd.read_csv(self.csv_file_path)

            #Selected data chart
            self.figureDataselected.clear()
            ax = self.figureDataselected.add_subplot(111)
            ax.set_facecolor((192/255,192/255,192/255,0.6))
            ColumnwithVal = ['PreHeatZone','HeatZone1','HeatZone2','SockingZone']
            Bardata = df.iloc[:, 1:5]
            Bardata.plot(kind='line', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.subplots_adjust(bottom=0.2)
            self.canvasDataselected.draw()

            #Rate of changes chart
            self.figureRateofchange.clear()
            ax = self.figureRateofchange.add_subplot(111)
            ax.set_facecolor((192/255,192/255,192/255,0.6))
            rate_of_change = df.diff() / df.shift()
            #rate_of_change = rate_of_change.iloc[1:]
            data = rate_of_change.iloc[:,1:5]
            data.plot(kind='line', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.subplots_adjust(bottom=0.2)
            self.canvasRateofchange.draw()

  
            


    def Predict(self):
        print('Predict button [pressed]')
        
        df = pd.read_csv(self.csv_file_path)
        df['FuelType'] = 0
        DeepLearnPrediction = PredDeeplearningModel(df)
        LinearRegPrediction = PredLinearRegressionModel(df)
        DecisTreePrediction = PredDecisionTreeModel(df)
        RandmForePrediction = PredRandomForestModel(df)
        TotalPrediction = [PredDeeplearningModel(df), PredLinearRegressionModel(df), PredDecisionTreeModel(df), PredRandomForestModel(df)]
        AveragePrediction = [sum(values) / len(TotalPrediction) for values in zip(*TotalPrediction)]

        data_dict = {
         'DeepLearnPrediction': DeepLearnPrediction,
        'LinearRegPrediction': LinearRegPrediction,
        'DecisTreePrediction': DecisTreePrediction,
        'RandmForePrediction': RandmForePrediction
        }
        PredictedDatadf = pd.DataFrame(data_dict)
        index_column = df['SrNo']
        #Prediction chart
        self.figurePredictionChart.clear()
        ax = self.figurePredictionChart.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        ax.set_xticks(range(len(index_column)))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        PredictedDatadf.plot(kind='line', ax=ax)
        plt.subplots_adjust(bottom=0.2)
        
        self.canvasPredictionChart.draw()

        #Prediction Range chart
        self.figurePredictionChart2.clear()
        ax = self.figurePredictionChart2.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        x = range(len(index_column))
        ax.set_xticks(range(len(index_column)))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        min_vals = np.minimum.reduce([DeepLearnPrediction, LinearRegPrediction, DecisTreePrediction, RandmForePrediction])
        max_vals = np.maximum.reduce([DeepLearnPrediction, LinearRegPrediction, DecisTreePrediction, RandmForePrediction])
        ax.fill_between(x, min_vals, max_vals, alpha=0.5)
        ax.plot(x,AveragePrediction, label='Average prediction',color='red')
        plt.subplots_adjust(bottom=0.2)
        self.canvasPredictionChart2.draw()
        
    def PredictCokeoven(self):
        print('Predict button [pressed]')
        
        df = pd.read_csv(self.csv_file_path)
        df['FuelType'] = 1
        df['AirFuelRatio'] = df['AirFuelRatio'] + 4
        DeepLearnPrediction = PredDeeplearningModel(df)
        LinearRegPrediction = PredLinearRegressionModel(df)
        DecisTreePrediction = PredDecisionTreeModel(df)
        RandmForePrediction = PredRandomForestModel(df)
        TotalPrediction = [PredDeeplearningModel(df), PredLinearRegressionModel(df), PredDecisionTreeModel(df), PredRandomForestModel(df)]
        AveragePrediction = [sum(values) / len(TotalPrediction) for values in zip(*TotalPrediction)]

        data_dict = {
         'DeepLearnPrediction': DeepLearnPrediction,
        'LinearRegPrediction': LinearRegPrediction,
        'DecisTreePrediction': DecisTreePrediction,
        'RandmForePrediction': RandmForePrediction
        }
        PredictedDatadf = pd.DataFrame(data_dict)
        #Prediction chart
        self.figurePredictionChart.clear()
        ax = self.figurePredictionChart.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        PredictedDatadf.plot(kind='line', ax=ax)
        plt.subplots_adjust(bottom=0.2)
        
        self.canvasPredictionChart.draw()

        #Prediction Range chart
        self.figurePredictionChart2.clear()
        ax = self.figurePredictionChart2.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        index_column = df['SrNo']
        x = range(len(index_column))
        ax.set_xticks(range(len(index_column)))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        min_vals = np.minimum.reduce([DeepLearnPrediction, LinearRegPrediction, DecisTreePrediction, RandmForePrediction])
        max_vals = np.maximum.reduce([DeepLearnPrediction, LinearRegPrediction, DecisTreePrediction, RandmForePrediction])
        ax.fill_between(x, min_vals, max_vals, alpha=0.5)
        ax.plot(x,AveragePrediction, label='Average prediction',color='red')
        plt.subplots_adjust(bottom=0.2)
        self.canvasPredictionChart2.draw()



        # Show the plot
        #plt.show()

class ThirdPage(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('Page3.ui', self)
        self.button2Mainpage = self.findChild(QtWidgets.QPushButton, 'bt_Mainpage')
        self.button2Mainpage.clicked.connect(self.GoMainpage)
        self.Lastdata = self.findChild(QtWidgets.QPushButton, 'bt_GetLastdata')
        self.Lastdata.clicked.connect(self.GetLastdata)
        self.ExeAnalysis = self.findChild(QtWidgets.QPushButton, 'Bt_Analysis')
        self.ExeAnalysis.clicked.connect(self.ExecAnalysis)

        self.containerLastdata = self.findChild(QtWidgets.QWidget, 'Wid_Last10Hrdata')
        self.figureLastdata = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasLastdata = FigureCanvas(self.figureLastdata)  
        layoutLastdata = QVBoxLayout(self.containerLastdata)  
        layoutLastdata.addWidget(self.canvasLastdata)

        self.containerConsPred = self.findChild(QtWidgets.QWidget, 'Wid_ConsVsPred')
        self.figureConsPred = plt.figure(facecolor=(192/255,192/255,192/255,0.6))  # Create a matplotlib figure
        self.canvasConsPred = FigureCanvas(self.figureConsPred)  
        layoutConsPred = QVBoxLayout(self.containerConsPred)  
        layoutConsPred.addWidget(self.canvasConsPred)

        

        
        
    def GoMainpage(self):
        self.Mainpage = MyApp()
        self.setCentralWidget(self.Mainpage)
    def GetLastdata(self):
        df = pd.read_csv('Pastdata.csv')
        
        #Selected data chart
        data1 = df.iloc[:, 1:5]
        data2 = df['GasConsumption']
        TotalTemp = [df['PreheatingZone'],df['HeatingZone1'],df['HeatingZone2'],df['SockingZone'] ]
        AverageTemp = pd.Series([sum(values) / len(TotalTemp) for values in zip(*TotalTemp)])
        data_dict = {
         'AverageTemp': AverageTemp,
        'ActualConsumption': data2
        }
        TempVsConsum = pd.DataFrame(data_dict)
       
        self.figureLastdata.clear()
        ax = self.figureLastdata.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        TempVsConsum.plot(kind='area', ax=ax, alpha=0.3)
        
        plt.subplots_adjust(bottom=0.2)

        self.canvasLastdata.draw()
    
    def ExecAnalysis(self):
        df = pd.read_csv('Pastdata.csv')
        index_column = df['SrNo']
        Actualdata = df['GasConsumption']
        df = df.iloc[:, :-1]
        DeepLearnPrediction = PredDeeplearningModel(df)
        LinearRegPrediction = PredLinearRegressionModel(df)
        DecisTreePrediction = PredDecisionTreeModel(df)
        RandmForePrediction = PredRandomForestModel(df)
        TotalPrediction = [PredDeeplearningModel(df), PredLinearRegressionModel(df), PredDecisionTreeModel(df), PredRandomForestModel(df)]
        AveragePrediction = [sum(values) / len(TotalPrediction) for values in zip(*TotalPrediction)]
        data_dict = {
         'PredictedConsumption': AveragePrediction,
        'ActualConsumption': Actualdata
        }
        PredictedActualDatadf = pd.DataFrame(data_dict)
        difference = PredictedActualDatadf['ActualConsumption'] - PredictedActualDatadf['PredictedConsumption']
        #Prediction chart
        self.figureConsPred.clear()
        ax = self.figureConsPred.add_subplot(111)
        ax.set_facecolor((192/255,192/255,192/255,0.6))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        #ax.set_xticks(range(len(index_column)))
        plotline = PredictedActualDatadf.plot(kind='line', ax=ax)
        second_line = plotline.get_lines()[1]  # Get the second line from the lines

        # Create a boolean mask where the difference is more than 200
        mask = np.abs(difference) > 100

        # Iterate over the line segments and set color and width individually
        for i in range(len(mask) - 1):
            color = 'red' if mask[i + 1] else 'Green'
            linewidth = 2.5 if mask[i + 1] else 1.5
            ax.plot(
                [i, i + 1],
                [PredictedActualDatadf['ActualConsumption'].iloc[i], PredictedActualDatadf['ActualConsumption'].iloc[i + 1]],
                color=color,
                linewidth=linewidth
            )

        plt.subplots_adjust(bottom=0.2)
        
        self.canvasConsPred.draw()





      










if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_app = MyApp()
    sys.exit(app.exec())


