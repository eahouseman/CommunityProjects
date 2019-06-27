"""
This is an attempt to translate speech recognition code originally written in R
to equivalent code in Python.  Note, however, that part of the solution involves
statistical algorithms for which R possesses the most sophisticated versions.
Consequently, this solution still relies on an R installation, 
accessed via the rpy2 library.  

This code is meant to be stepped through in small blocks

"""
###########
# Set working directory
MYPATH = "C:\\LocalDocuments\\Projects\\PortlandDataScience\\"

# Load utility functions
with open('speech-recognition-utils.py') as fd:
    exec(fd.read())

###########
# Read first training set
train1 =readWaveFiles(MYPATH+"numbers\\numbers")

yTrain1 = []  # Classification for training set
for x in train1['names']:
     yTrain1.append(x[0:2])

###########
# Read second training set
train2 =readWaveFiles(MYPATH+"morenumbers\\numbers")
yTrain2 = []  # Classification for training set
for x in train2['names']:
     yTrain2.append(x[0:2])

trainAll = {'data':train1['data']+train2['data'], 'names':train1['names']+train2['names']} 
yTrainAll = yTrain1 + yTrain2

###########
# Examples of plotting and envelope calculation
plotRawWave(train1['data'][0], clip1=500, clip2=500)

ex1 = getSmoothEnvelope(train1['data'][0])
plotRawWaveAndEnvelope(Timeseries(train1['data'][0],absolutevalue), ex1)

###########
ex2 = getSmoothEnvelope(train1['data'][1])
plotRawWaveAndEnvelope(Timeseries(train1['data'][1],absolutevalue), ex2)

ex3 = getSmoothEnvelope(train1['data'][2])
plotRawWaveAndEnvelope(Timeseries(train1['data'][2],absolutevalue), ex3)

closeRPlot()

################ Preprocessing
#  IMPORTANT:  note the difference between Block 1 and Block 2 below!

########### Block1
# First time:  calculating the envelopes (takes about 15 to 30 minutes)
rs_env = []
for x in trainAll['data']:
   rs_env.append(getSmoothEnvelope(x))

# Save output to JSON so accessible in the future
jsonOut = []
for x in rs_env:
   jsonOut.append(x.data)

import json
with open('gam-envelopes-190626.json', 'w') as outfile:  
   json.dump(jsonOut, outfile)

########### Block2
# For future sessions, read from saved JSON file
import json
jsonIn = json.load(open('gam-envelopes-190626.json', 'r'))
rs_env = []
for i in range(len(trainAll['data'])):
   ts = Timeseries(trainAll['data'][i])
   rs_env.append(Timeseries(jsonIn[i], clip1=0, clip2=0, time=ts.time))

########### Now get the envelope statistics
# Get the peak envelope stats for all files
rs_stat = []
rs_flag = []
for x in rs_env:
    try:
      s = getEnvelopePeakStats(x)
      rs_stat.append(s)
      rs_flag.append(1)
    except:
      rs_flag.append(0)
      pass

# Number of errors (should be 4)
len(rs_flag)-sum(rs_flag)
