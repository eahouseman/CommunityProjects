import os
import math
import numpy
#
from scipy.io import wavfile
#
os.putenv('R_USER',"C:\Python36\Lib\site-packages\rpy2")
import rpy2
import rpy2.robjects
#
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
mgcv = importr("mgcv")

def identityfunction(x):
   out = []
   for i in range(len(x)):
      out.append(x[i])
   return(out)

def absolutevalue(x):
   out = []
   for i in range(len(x)):
      out.append(abs(x[i]))
   return(out)

def difference(x):
   out = []
   for i in range(1,len(x)):
      out.append(x[i]-x[i-1])
   return(out)

def readWaveFiles(dir):
    ff = os.listdir(dir)
    nf = len(ff)
    out = []
    for i in range(nf):
        rate, sound = wavfile.read(dir+'\\'+ff[i])
        data = []
        if sound.dtype=='int16':
            for j in range(len(sound)):
                data.append(sound[j]/32768)
        elif sound.dtype=='int8':
            for j in range(len(sound)):
                data.append(sound[j]/128 - 1)
        else:
            for j in range(len(sound)):
                data.append(sound[j])
        out.append(data)
    return({'data' : out, 'names' : ff})

class Timeseries:
    def __init__(self, data, transform=identityfunction, clip1=500, clip2=500, time=None):
       nx = len(data)
       xs = data[slice(clip1,nx-clip2)]
       if time is None:
           tt = list(range(clip1,nx-clip2))
       else:
           tt = time
       self.time = tt
       self.data = transform(xs)

def getSmoothEnvelope(wvdata, clip1=500, clip2=500):
    ts = Timeseries(wvdata, absolutevalue, clip1, clip2)
    #
    r = rpy2.robjects
    y = r.FloatVector(ts.data)
    x = r.FloatVector(ts.time)
    #
    r.globalenv["y"] = y
    r.globalenv["x"] = x
    #
    ff = Formula("y~s(x)")
    gf = mgcv.gam(ff)
    yhat = numpy.array(r.r.predict(gf))
    #
    return(Timeseries(yhat, clip1=0, clip2=0, time=ts.time))

def plotRawWave(wvdata, transform=identityfunction, clip1=500, clip2=500):
    ts = Timeseries(wvdata, transform, clip1, clip2)
    rpy2.robjects.r.plot(ts.time, ts.data, type='l', xlab='time', ylab='amplitude')

def plotRawWaveAndEnvelope(ts, envelope):
    rpy2.robjects.r.plot(ts.time, ts.data, type='l', xlab='time', ylab='amplitude')
    rpy2.robjects.r.lines(envelope.time, envelope.data, col='red')
    rpy2.robjects.r['dev.flush']()

def closeRPlot():
    rpy2.robjects.r['dev.off']()

def getEnvelopePeakStats(envelope, errorClip=5000):
   # Get time corresponding to maximum envelope
   nx = len(envelope.data)
   x0 = numpy.argsort(envelope.data)[nx-1]
   #
   # If maximum occurs at the beginning, clip some more from the beginning
   if x0==0:
       x0=numpy.argsort(envelope.data[slice(errorClip,nx)])[nx-errorClip-1]+errorClip
   #
   y0 = envelope.data[x0]
   pk = envelope.time[x0]
   #
   d1 = difference(envelope.data)
   d2 = difference(d1)
   a0 = d2[x0-1]
   #
   # Half-width calculated as the difference between peak and x-intercepts
   # of the parabola implied by the 2nd difference and the peak height
   halfwidth = math.sqrt(-y0/a0) 
   return({'peakheight':y0,'peak':pk,'curvature':a0, 'peakwidth':halfwidth})
