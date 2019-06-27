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

################ Empirical Bayes Fitting

########### Get the weighted Fast Fourier Transform data
rs_fft = []
rs_fft_ix = []
ii = -1
for i in range(len(rs_flag)):
    if rs_flag[i]==1:
        ii = ii+1
        print(i,ii)
        x = Timeseries(trainAll['data'][i],absolutevalue)
        rs_fft.append(getWeightedFFT(x, mu=rs_stat[ii]["peak"], sigma=rs_stat[ii]["peakwidth"], 
            scale=rs_stat[ii]["peakheight"], sigmaMult=5, K=15))
        rs_fft_ix.append(i)

########### Get some indexes for associating data elements
rs_fft_ixrev = numpy.repeat(0,len(yTrainAll))
for i in range(len(rs_fft_ix)):
   rs_fft_ixrev[ rs_fft_ix[i] ] = i

ixTrainAll = indexSet(rs_fft_ix, yTrainAll)

########### Get multivariate normal sufficient statistics
rs_mvn = {}
for k in [*ixTrainAll]:
     rs_mvn[k] = getMVNStats(rs_fft,ixTrainAll[k],rs_fft_ixrev)

########### Get Empirical Bayes posterior probabilities
rs_eb = []
for i in range(len(rs_fft)):
  rs_eb.append(eBayes(rs_fft[i], rs_mvn))
 
########### Classify by max Empirical Bayes posterior probability
rs_ebClass = []
rs_ebClassCorrect = []
for i in range(len(rs_fft)):
   keys = [*rs_eb[i]]
   mx = keys[0]
   for k in keys:
      if rs_eb[i][k]>rs_eb[i][mx]:
          mx = k
   rs_ebClass.append(mx) 
   rs_ebClassCorrect.append(1*(mx==yTrainAll[rs_fft_ix[i]]))

########### Assess classification
# Correctly classified (naive, not cross-validated)
sum(rs_ebClassCorrect)/len(rs_ebClassCorrect)

########### Show heatmap of classification
PALETTE = ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628",
   "#8DD3C7","#FFFFB3","#BEBADA","#FB8072","#80B1D3","#FDB462")

trainAll_annot = ["" for x in range(len(trainAll['data']))]
k = [*ixTrainAll]
for j in range(len(k)):
   for i in range(len(ixTrainAll[k[j]])):
      trainAll_annot[ixTrainAll[k[j]][i]] = PALETTE[j]

rs_class_annot = ["" for x in range(len(rs_fft_ix))]
for i in range(len(rs_fft_ix)):
   rs_class_annot[i] = trainAll_annot[rs_fft_ix[i]]

showHeatmap(rs_eb,rs_class_annot)
closeRPlot()
