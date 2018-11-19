'''
Created on Oct 31, 2018

@author: mohame11
'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from learn_MMmK import *
from simple_NN import *
from sliding_NN import *
from LSTM import *
import numpy as np

class Tester(object):
    '''
    classdocs
    '''


    def __init__(self, model = None, direct = '', fname = '', scalingThreshold = 0.005, calls_to_packets = 1.0, interval = 1):
        self.model = model
        self.dir = direct
        self.fname = fname
        self.scalingThreshold = scalingThreshold
        self.calls_to_packets = calls_to_packets
        self.interval = interval
        
    
    
    def parseDataFile(self, fpath, inputPacketsCols, droppedPacketsCols):
        df = pd.read_csv(fpath, usecols = inputPacketsCols+droppedPacketsCols)
        df.fillna(0, inplace=True) # replace missing values (NaN) to zero
        return df
    
    
    def plot_est_empPK_batch(self):
        #using client data
        fpath = self.dir + self.fname
        inputPacketsCols = ['CallRate(P)']
        droppedPacketsCols = ['FailedCall(P)']
        interval = self.interval
        calls_to_packets = self.calls_to_packets
        scalingThreshold = self.scalingThreshold
        
        df = self.parseDataFile(fpath, inputPacketsCols, droppedPacketsCols)
        
        
        totalInputPackets = 0
        totalDroppedPackets = 0
        MSE = 0
        intervalCount = 0
        print 'time_interval empirical_PK estimated_PK squareLoss'
        est_PKs = []
        emp_PKs = []
        my_lambdas = []
        timeIntervals = []
        squaredLoss = 0
        skipFlag = False
        for i, row in df.iterrows():
            
            intervalCount += 1
            totalInputPackets = 0
            totalDroppedPackets = 0
          
            for inpCol in inputPacketsCols:
                totalInputPackets += row[inpCol] * calls_to_packets
                
            for dropCol in droppedPacketsCols:
                totalDroppedPackets += row[dropCol] * calls_to_packets
                
            mylmabda = totalInputPackets
            try:
                empirical_PK = float(totalDroppedPackets) / totalInputPackets
                
            except:
                skipFlag = True
                
            if (totalDroppedPackets == 0 and totalInputPackets == 0) or totalInputPackets == 0 or empirical_PK > 1 or skipFlag:
                skipFlag = False
                continue
            my_lambdas.append([mylmabda])
            emp_PKs.append(empirical_PK)
            timeIntervals.append(intervalCount)
            
            
        my_lambdas.append(-1)
        #emp_PKs.append(-1)
        batch_X, batch_Y = batchfiy(my_lambdas, emp_PKs)
        batch_X = batch_X[0]
        batch_Y = batch_Y[0]
        pred = self.model.predict(batch_X)
        pred = torch.clamp(pred, min = 0, max = 1)
        est_PKs = list(pred.data.numpy())
        
        
        fig = plt.figure(1, figsize=(6, 4))
        #plt.xticks([x for x in range(1,max(timeIntervals))])
        axes = plt.gca()
        ax = plt.axes()
        #axes.set_ylim([-0.05,0.6])
        
        
        #drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Time Interval(1 interval=' +str(interval)+' s)')
        lines = plt.plot(timeIntervals, est_PKs, '--r' ,label='Estimated PK') 
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(timeIntervals, emp_PKs, 'b' ,label='Empirical PK') 
        plt.setp(lines, linewidth=2.0)
        
        #finding the time for scaling
        scalingTime = -1
        criticalPK = 0
        for i in range(len(est_PKs)):
            if est_PKs[i] > scalingThreshold: 
                scalingTime = timeIntervals[i]
                criticalPK = est_PKs[i]
                break
        
        max_y = max(max(est_PKs), max(emp_PKs))
        y = list(np.arange(0.0, max_y, max_y/5))
        ax.text(scalingTime, max(y), 't='+str(scalingTime)+'\nProb Thresh.='+str(scalingThreshold), fontsize=12)
        lines = plt.plot([scalingTime for t in y], np.arange(0.0, max_y, max_y/5), ':g' ,label='Scaling Time') 
        plt.setp(lines, linewidth=2.0)
        
        plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
        fig.suptitle(fname, fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()                                                                     
        #plt.savefig(resultsPath+'combined_rec_prec_plot_withActionSampling.pdf', bbox_inches='tight')
        plt.show() 
        
    
    def plot_estPK_empPK(self):
        #using client data
        fpath = self.dir + self.fname
        inputPacketsCols = ['CallRate(P)']
        droppedPacketsCols = ['FailedCall(P)']
        interval = self.interval
        calls_to_packets = self.calls_to_packets
        scalingThreshold = self.scalingThreshold
        
        df = self.parseDataFile(fpath, inputPacketsCols, droppedPacketsCols)
        
        if interval == -1:
            interval = df.shape[0]-1
        
        totalInputPackets = 0
        totalDroppedPackets = 0
        MSE = 0
        intervalCount = 0
        print 'time_interval empirical_PK estimated_PK squareLoss'
        est_PKs = []
        emp_PKs = []
        my_lambdas = []
        timeIntervals = []
        squaredLoss = 0
        skipFlag = False
        for i, row in df.iterrows():
            if i % interval == 0 and i != 0:
                intervalCount += 1
                if intervalCount == 134:
                    dbg = 1
                
                avgDroppedPackets = totalDroppedPackets / float(interval)
                avgInputPackets   = totalInputPackets / float(interval)
                
                my_lambda = avgInputPackets
                my_lambdas.append(my_lambda)
                
                try:
                    empirical_PK = float(totalDroppedPackets) / totalInputPackets
                except:
                    skipFlag = True
                    
                if (totalDroppedPackets == 0 and totalInputPackets == 0) or totalInputPackets == 0 or empirical_PK > 1:
                    #empirical_PK = 0
                    skipFlag = True
                    
                
                try:
                    estimated_PK = self.model.predict(my_lambda, clampNumbers = True)
                    
                except:
                    skipFlag = True
                    
                if skipFlag:
                    skipFlag = False
                    totalInputPackets = 0
                    totalDroppedPackets = 0
                   
                    for inpCol in inputPacketsCols:
                        totalInputPackets += row[inpCol] * calls_to_packets
                        
                    for dropCol in droppedPacketsCols:
                        totalDroppedPackets += row[dropCol] * calls_to_packets
                    
                    continue
                
                
                    
                     
                timeIntervals.append(intervalCount)
                emp_PKs.append(empirical_PK)
                est_PKs.append(estimated_PK)  
                
                squaredLoss = (empirical_PK - estimated_PK)**2
                MSE += squaredLoss
                
                print intervalCount, empirical_PK, estimated_PK, squaredLoss
                
                totalInputPackets = 0
                totalDroppedPackets = 0
               
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol] * calls_to_packets
                    
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol] * calls_to_packets
               
                
            else:
                
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol] * calls_to_packets
                    
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol] * calls_to_packets
        
        print 'MSE=', squaredLoss/intervalCount
        
        fig = plt.figure(1, figsize=(6, 4))
        #plt.xticks([x for x in range(1,max(timeIntervals))])
        axes = plt.gca()
        ax = plt.axes()
        #axes.set_ylim([-0.05,0.6])
        
        
        #drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Time Interval(1 interval=' +str(interval)+' s)')
        lines = plt.plot(timeIntervals, est_PKs, '--r' ,label='Estimated PK') 
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(timeIntervals, emp_PKs, 'b' ,label='Empirical PK') 
        plt.setp(lines, linewidth=2.0)
        
        #finding the time for scaling
        scalingTime = -1
        criticalPK = 0
        for i in range(len(est_PKs)):
            if est_PKs[i] > scalingThreshold: 
                scalingTime = timeIntervals[i]
                criticalPK = est_PKs[i]
                break
        
        max_y = max(max(est_PKs), max(emp_PKs))
        y = list(np.arange(0.0, max_y, max_y/5))
        ax.text(scalingTime, max(y), 't='+str(scalingTime)+'\nProb Thresh.='+str(scalingThreshold), fontsize=12)
        lines = plt.plot([scalingTime for t in y], np.arange(0.0, max_y, max_y/5), ':g' ,label='Scaling Time') 
        plt.setp(lines, linewidth=2.0)
        
        plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
        fig.suptitle(fname, fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()                                                                     
        #plt.savefig(resultsPath+'combined_rec_prec_plot_withActionSampling.pdf', bbox_inches='tight')
        plt.show() 
        
    def getData(self, summaryFile = 'summary_data_dump.csv', minDropRate = 0, maxDropRate = 1e100):
        sfile = self.dir + summaryFile
        inputPacketsCols = ['CallRate(P)']
        droppedPacketsCols = ['FailedCall(P)']
        
        df = pd.read_csv(sfile, usecols = ['Rate File', ' Failed Calls'])
        df.fillna(0, inplace=True)
        train_X = []
        train_Y = [] 
        for i, row in df.iterrows():
            if row[' Failed Calls'] < minDropRate or row[' Failed Calls'] > maxDropRate:
                continue
            fname = 'sipp_data_' + row['Rate File'] + '_1.csv'
            simulationFile = self.dir + fname #sipp_data_UFF_Perdue_01_1_reduced_1.csv     UFF_Perdue_01_12_reduced
            
            curr_df = pd.read_csv(simulationFile, usecols = inputPacketsCols+droppedPacketsCols)
            curr_df.fillna(0, inplace=True) # replace missing values (NaN) to zero
            for j, curr_row in curr_df.iterrows():
                try:
                    the_lambda = float(curr_row['CallRate(P)'])
                    failed = float(curr_row['FailedCall(P)'])
                    if failed > the_lambda:
                        continue
                    PK = failed/the_lambda
                except:
                    continue
                train_X.append(the_lambda)
                train_Y.append(PK)
                
        return train_X, train_Y
    
    def cal_MSE(self, data_X, data_Y):
        squaredLoss = 0.0
        for i in range(len(data_X)):
            pred = self.model.predict(data_X[i], clampNumbers = True)
            squaredLoss += (pred - data_Y[i])**2
        
        squaredLoss /= len(data_X)
        return squaredLoss
            
        
        


if __name__ == "__main__":
    mpl.rcParams.update({'font.size': 17})
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_restricted'
    modelName = 'LSTM_model'
    #model = torch.load('simpleNN_model')
    #model = torch.load('simpleNN_model')
    #model = torch.load('sliding_model')
    #model = torch.load('MMmK_model_1_25_812')
    model = torch.load(modelName)
    direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/'
    
    #direct += '/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    #direct += '/results_CORES_2_K_DEFT_SCALE_43_2018.10.29-18.56.45/sipp_results/'
    #direct += '/results_CORES_3_K_DEFT_SCALE_64_2018.10.29-20.38.11/sipp_results/'
    direct += '/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    
    #fname = 'sipp_raw_data_UFF_Perdue_07_7_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_02_10_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_08_14_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_01_13_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_04_23_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_02_29_reduced_1.csv'
    fname = 'sipp_raw_data_UFF_Perdue_04_42_reduced_1.csv'
    
    
    t = Tester(model = model, direct = direct, fname = fname, scalingThreshold = 0.005, calls_to_packets = 1.0, interval = 1)
    
    #t.plot_estPK_empPK()
    t.plot_est_empPK_batch()
    
    '''
    data_X, data_Y = t.getData(summaryFile='summary_data_dump.csv', minDropRate = 0, maxDropRate = 1e100)
    MSE = t.cal_MSE(data_X, data_Y)
    print 'dir=', direct
    print 'Model Name = ', modelName
    print '#test samples=',len(data_X)
    print 'MSE=', MSE
    '''
    
    
    
    