#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 

import numpy as np
import matplotlib.pyplot as plt
import csv
 
#data=np.genfromtxt(r"data.csv", skip_header=1,delimiter=",")
print ("The data set has the following features: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP & Classification")

print('======================================================')
#
 
testing=[]
training=[]
testingLabel=[] 
trainingLabel=[]
#summary all the feature, report the minimum, maximum, average, median of each feature

with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<0.25:
            testing.append(row['Age'])
        elif random >=0.25:
            training.append(row['Age'])
    testingLabel.append([])
    testingLabel[0].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[0].append(np.array(training).astype(np.float))
        
    print ('the 1st col is Age')
    print('For the testing set, the max is',np.max(testingLabel[0]),' the min is ',np.min(testingLabel[0]),' the median is ',np.median(testingLabel[0]),' the average is',np.average(testingLabel[0]))
    
    print('For the training set, the max is',np.max(trainingLabel[0]),' the min is ',np.min(trainingLabel[0]),' the median is ',np.median(trainingLabel[0]),' the average is',np.average(trainingLabel[0]))
    
    print('======================================================')
testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<0.25:
            testing.append(row['BMI'])
        elif random >=0.25:
            training.append(row['BMI'])    
    testingLabel.append([])
    testingLabel[1].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[1].append(np.array(training).astype(np.float))
    print ('the 2nd col is BMI')
    print('For the testing set, the max is',np.max(testingLabel[1]),' the min is ',np.min(testingLabel[1]),' the median is ',np.median(testingLabel[1]),' the average is',np.average(testingLabel[1]))
    
    print('For the training set, the max is',np.max(trainingLabel[1]),' the min is ',np.min(trainingLabel[1]),' the median is ',np.median(trainingLabel[1]),' the average is',np.average(trainingLabel[1]))
    print('======================================================')
    
testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['Glucose'])
        elif random >0.25:
            training.append(row['Glucose'])    
    testingLabel.append([])
    testingLabel[2].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[2].append(np.array(training).astype(np.float))
    print ('the 3rd col is Glucose')
    print('For the testing set, the max is',np.max(testingLabel[2]),' the min is ',np.min(testingLabel[2]),' the median is ',np.median(testingLabel[2]),' the average is',np.average(testingLabel[2]))
    
    print('For the training set, the max is',np.max(trainingLabel[2]),' the min is ',np.min(trainingLabel[2]),' the median is ',np.median(trainingLabel[2]),' the average is',np.average(trainingLabel[2]))
    print('======================================================')
     
testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['Insulin'])
        elif random >0.25:
            training.append(row['Insulin'])    
    testingLabel.append([])
    testingLabel[3].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[3].append(np.array(training).astype(np.float))
    print ('the 4th col is Insulin')
    print('For the testing set, the max is',np.max(testingLabel[3]),' the min is ',np.min(testingLabel[3]),' the median is ',np.median(testingLabel[3]),' the average is',np.average(testingLabel[3]))
    
    print('For the training set, the max is',np.max(trainingLabel[3]),' the min is ',np.min(trainingLabel[3]),' the median is ',np.median(trainingLabel[3]),' the average is',np.average(trainingLabel[3]))
    print('======================================================')
     
  
testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['HOMA'])
        elif random >0.25:
            training.append(row['HOMA'])    
    testingLabel.append([])
    testingLabel[4].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[4].append(np.array(training).astype(np.float))
    print ('the 5th col is HOMA')
    print('For the testing set, the max is',np.max(testingLabel[4]),' the min is ',np.min(testingLabel[4]),' the median is ',np.median(testingLabel[4]),' the average is',np.average(testingLabel[4]))
    
    print('For the training set, the max is',np.max(trainingLabel[4]),' the min is ',np.min(trainingLabel[4]),' the median is ',np.median(trainingLabel[4]),' the average is',np.average(trainingLabel[4]))
    print('======================================================')
     
  
testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['Leptin'])
        elif random >0.25:
            training.append(row['Leptin'])    
    testingLabel.append([])
    testingLabel[5].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[5].append(np.array(training).astype(np.float))
    print ('the 6th col is Leptin')
    print('For the testing set, the max is',np.max(testingLabel[5]),' the min is ',np.min(testingLabel[5]),' the median is ',np.median(testingLabel[5]),' the average is',np.average(testingLabel[5]))
    
    print('For the training set, the max is',np.max(trainingLabel[5]),' the min is ',np.min(trainingLabel[5]),' the median is ',np.median(trainingLabel[5]),' the average is',np.average(trainingLabel[5]))
    print('======================================================')
     
testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['Adiponectin'])
        elif random >0.25:
            training.append(row['Adiponectin'])    
    testingLabel.append([])
    testingLabel[6].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[6].append(np.array(training).astype(np.float))
    print ('the 7th col is Adiponectin')
    print('For the testing set, the max is',np.max(testingLabel[6]),' the min is ',np.min(testingLabel[6]),' the median is ',np.median(testingLabel[6]),' the average is',np.average(testingLabel[6]))
    
    print('For the training set, the max is',np.max(trainingLabel[6]),' the min is ',np.min(trainingLabel[6]),' the median is ',np.median(trainingLabel[6]),' the average is',np.average(trainingLabel[6]))
    print('======================================================')
         

testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['Resistin'])
        elif random >0.25:
            training.append(row['Resistin'])    
    testingLabel.append([])
    testingLabel[7].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[7].append(np.array(training).astype(np.float))
    print ('the 8th col is Resistin')
    print('For the testing set, the max is',np.max(testingLabel[7]),' the min is ',np.min(testingLabel[7]),' the median is ',np.median(testingLabel[7]),' the average is',np.average(testingLabel[7]))
    
    print('For the training set, the max is',np.max(trainingLabel[7]),' the min is ',np.min(trainingLabel[7]),' the median is ',np.median(trainingLabel[7]),' the average is',np.average(trainingLabel[7]))
    print('======================================================')
         

testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['MCP'])
        elif random >0.25:
            training.append(row['MCP'])    
    testingLabel.append([])
    testingLabel[8].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[8].append(np.array(training).astype(np.float))
    print ('the 9th col is MCP')
    print('For the testing set, the max is',np.max(testingLabel[8]),' the min is ',np.min(testingLabel[8]),' the median is ',np.median(testingLabel[8]),' the average is',np.average(testingLabel[8]))
    
    print('For the training set, the max is',np.max(trainingLabel[8]),' the min is ',np.min(trainingLabel[8]),' the median is ',np.median(trainingLabel[8]),' the average is',np.average(trainingLabel[8]))
    print('======================================================')
       
testing=[]
training=[]  
with open ("data.csv") as file:
    reader= csv.DictReader(file, delimiter=",")
    for row in reader:
#        print (row)
        random = np.random.uniform(0,1)
        if random<=0.25:
            testing.append(row['Classification'])
        elif random >0.25:
            training.append(row['Classification'])    
    testingLabel.append([])
    testingLabel[9].append(np.array(testing).astype(np.float))
    trainingLabel.append([])
    trainingLabel[9].append(np.array(training).astype(np.float))
    print ('the 10th col is Classification')
    print('For the testing set, the max is',np.max(testingLabel[9]),' the min is ',np.min(testingLabel[9]),' the median is ',np.median(testingLabel[9]),' the average is',np.average(testingLabel[9]))
    
    print('For the training set, the max is',np.max(trainingLabel[9]),' the min is ',np.min(trainingLabel[9]),' the median is ',np.median(trainingLabel[9]),' the average is',np.average(trainingLabel[9]))
    print('======================================================')
            
  
    
    # three graphes:
    #graph 1: Age VS BMI
    trainingLabel[0]= np.array(trainingLabel[0])
    trainingLabel[1]= np.array(trainingLabel[1])
    plt.scatter(trainingLabel[1],trainingLabel[1],color="red",marker=".")
    plt.scatter(trainingLabel[0],trainingLabel[0],color="yellow",marker=".")
    plt.xlabel('BMI')
    plt.ylabel('Age')
    plt.title('Age VS BMI')
    plt.show()
    
     #graph 1: MI VS Classification'
    trainingLabel[9]= np.array(trainingLabel[9])
    trainingLabel[1]= np.array(trainingLabel[1])
    plt.scatter(trainingLabel[1],trainingLabel[1],color="blue",marker="*")
    plt.scatter(trainingLabel[9],trainingLabel[9],color="red",marker="*")
    plt.xlabel('Classification')
    plt.ylabel('BMI')
    plt.title('BMI VS Classification')
    plt.show()
    
     #graph 1: Insulin VS Glucose
    trainingLabel[2]= np.array(trainingLabel[2])
    trainingLabel[3]= np.array(trainingLabel[3])
    plt.scatter(trainingLabel[3],trainingLabel[3],color="green",marker="+")
    plt.scatter(trainingLabel[2],trainingLabel[2],color="pink",marker="+")
    plt.xlabel('Insulin')
    plt.ylabel('Glucose')
    plt.title('Insulin VS Glucose')
    plt.show()
    
    #graph 3 : bar chat, the max of each feature
    times=[np.max(trainingLabel[0]),np.max(trainingLabel[1]),np.max(trainingLabel[2]),np.max(trainingLabel[3]),np.max(trainingLabel[4]),np.max(trainingLabel[5]),np.max(trainingLabel[6]),np.max(trainingLabel[7]),np.max(trainingLabel[8]) ] 
    title = ["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP"]
    plt.bar(title,times)
    plt.title("the maximum of each feature")
    plt.show()
    




















    

