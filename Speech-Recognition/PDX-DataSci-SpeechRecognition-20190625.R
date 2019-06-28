################################################
# Strategy
# I.  Remove spurious amplitude peaks
#   A.  Use GAM to smooth absolute amplitude, producing peaks
#   B.  Get center and width of highest peak
# 
# II.  Fit (weighted) Fast-Fourier-Transform (FFT)
#   A.  Use peak center to translate waveform so that time zero corresponds to peak
#   B.  Use center and width to obtain case weights (based on Gaussian distance from peak)
#   C.  Use width to determine length parameter of FFT
#   D.  Transform centered data to absolute amplitude (instead of signed amplitude)
#   E.  Get Fourier coefficients (15 harmonics seems sufficient for this problem)
#
# III.  Build Naive Bayes Classifier  
#   A.  Get multivariate normal stats for each (true) number spoken
#   B.  Form the prediction of any set of FFT coefficients as the posterior probability
#       of class assigment, with the stats in A forming the prior distribution
#       (Note:  each number is considered equally probable in the prior distribution).

################################################

################ Initialization

### Load libraries
library(sound)
library(RColorBrewer)
library(randomForest)
library(mgcv)

# Set working directory
setwd("C:\\LocalDocuments\\Projects\\PortlandDataScience")

################ Function definitions

# Log Density of multivariate normal (MVN) distribution
dmvnorm = function(x, mu, Sigma){
   r = x-mu
   eigv = eigen(Sigma, only.values=TRUE)$val
   -(log(2*pi)*length(mu)+sum(log(abs(eigv)))+r%*%solve(Sigma,r))/2
}

# Emprical Bayes on MVN data (equiprobable classes)
ebayes = function(x, muSigList){
  sapply(muSigList, function(ms)dmvnorm(x,ms$mu, ms$Sigma))
}

# Read .wav files from a directory
readWaveFiles = function(dir='numbers\\numbers'){
  dr = getwd()
  setwd(dir)
  ff = list.files()
  nf = length(ff)
  wavs = list()

  for(i in 1:nf){
    wavs[[i]] = loadSample(ff[i])
  }
  names(wavs) = ff
  setwd(dr)

  wavs
}

# Create an object that clips .wav data and stores as time series
newTimeSeries = function(wv, transform=function(x)x, clip1=500, clip2=500){
  xx = as.vector(wv$sound)
  nx = length(xx)
  ix = (1:nx)[-c(1:clip1,nx-(1:clip2)+1)]
  list(`time`=ix-1, `data`=transform(xx[ix]))
}

# Plot raw time series data
plotRawWave = function(ts, transform=function(x)x, clip1=500, clip2=500,...){
  xx = ts$data
  tt = ts$time
  plot(tt, transform(xx), type='l', xlab='time', ylab='amplitude',...)
}


# Use generalized additive model (GAM) to calculate a smooth envelope
# Note:  the statistically proper approach would be to use a quasipoisson model
#   but due to computational costs of nonlinear models, the Gaussian model is
#   being used as a decent approximation.
getSmoothEnvelope = function(ts){
  y = ts$data
  x = ts$time
  gf = gam(y~s(x))
  predict(gf)
}

# Get peak stats from GAM-based envelope
getEnvelopePeakStats = function(ts, env, errorClip=5000){
  i0 = which.max(env) # Get index corresponding to maximum envelope

  # If maximum occurs at the beginning, clip some more from the beginning
  if(i0==1) {
     i0=which.max(env[-(1:errorClip)])+errorClip
     cat('\tclipped\n')
  }
  y0 = env[i0]
  d1 = diff(env)  # First difference
  d2 = diff(d1)  # Second difference
  a0 = d2[i0-1]
  
  # Half-width calculated as the difference between peak and x-intercepts
  # of the parabola implied by the 2nd difference and the peak height
  halfwidth = sqrt(-y0/a0) 

  out = c(`peakheight`=y0,`peak`=ts$time[i0],`curvature`=a0,`peakwidth`=halfwidth)
  names(out) = c('peakheight','peak','curvature','peakwidth')
  out
}

# Weighted Fast-Fourier-Transform (general function)
FFT = function(t,x,K=10,L=length(t),weight=rep(1,length(t)),saveBasis=FALSE,ridge=0){
  tt = 2*pi*t/L
  ss = sin(outer(tt, 1:K, '*'))[,K:1]
  cc = cos(outer(tt, 0:K, '*'))
  B = cbind(ss,cc)
  colnames(B) = c(-(K:1),0,1:K)
  Bt = t(weight*B)
  BtB = Bt%*%B
  if(ridge>0) diag(BtB) = diag(BtB)+ridge
  out = list(coef = solve(BtB, Bt%*%as.vector(x)))
  rownames(out$coef) = colnames(B)

  if(saveBasis) out$basis = B
  out
}

# Weighted FFT for this specific problem (inputs: wave, center, width, etc.)
getWeightedFFT = function(ts, mu, sigma, scale, sigmaMult=3, K=10, ridge=1e-6){
  xx = ts$data/scale
  nx = length(xx)
  tt = ts$time-mu

  sigma2=sigma*sigma
  FFT(tt,ts$data/scale,K=K,L=sigmaMult*sigma,weight=exp(-tt*tt/sigma2/2),ridge=ridge)
}

# Create a randomly generated partion for K-fold cross-validation
randomlyPartition = function(i, folds=10){
  split(i, sample(1:folds, length(i), replace=TRUE))
}

################ Read data

# Palette for classes
palette = c(brewer.pal(7,'Set1'), brewer.pal(6,'Set3'))

# Read in training data set #1
train1 = readWaveFiles()
yTrain1 = substring(names(train1),1,2)
ixTrain1 = split(1:length(yTrain1), yTrain1)

# Read in training data set #2
train2 = readWaveFiles('moreNumbers\\numbers')
yTrain2 = substring(names(train2),1,2)
ixTrain2 = split(1:length(yTrain2), yTrain2)

# Combine training sets
trainAll = c(train1, train2)
yTrainAll = c(yTrain1, yTrain2)
ixTrainAll = split(1:length(yTrainAll), yTrainAll)

# Create clipped time series objects
tsAll = lapply(trainAll, newTimeSeries)
tsAbs = lapply(trainAll, newTimeSeries, transform=abs)

################ Preprocessing

if(!file.exists('gam-envelopes-190627.rds')) {
  # First time:  get smooth envelopes and save (due to length of time to process)
  rs.env = list()
  for(i in 1:length(tsAbs)){
    cat(i,'\n')
    rs.env[[i]] = getSmoothEnvelope(tsAbs[[i]])
    plotRawWave(tsAbs[[i]])
    lines(tsAbs[[i]]$time, rs.env[[i]], col='red')
  }

  saveRDS(rs.env, file='gam-envelopes-190627.rds')
}

if(file.exists('gam-envelopes-190627.rds')){
  # Later times (after smooth envelopes have been saved)
  rs.env = readRDS('gam-envelopes-190627.rds')
}

# Get the peak envelope stats for all files
rs.stat = t(mapply(getEnvelopePeakStats, ts=tsAbs, env=rs.env))

# Create a matrix of FFT coefficients
rs.fftList = mapply(getWeightedFFT, ts=tsAbs, 
  mu=rs.stat[,"peak"], sigma=rs.stat[,"peakwidth"], scale=rs.stat[,"peakheight"], 
  sigmaMult=5, K=15, SIMPLIFY=FALSE)
rs.fft = t(sapply(rs.fftList, function(u)u$coef))
colnames(rs.fft) = rownames(rs.fftList[[1]]$coef)

# filter errors
flag = !apply(is.na(rs.fft),1,any) 
sum(!flag) # How many errors?

# Kruskal-Wallis statistics for each coefficient (across all classes)
kruskal.stats = apply(rs.fft[flag,],2,function(u)kruskal.test(u~factor(yTrainAll[flag]))$p.value)
sort(kruskal.stats)

# View clustering heatmap of coefficients
heatmap(rs.fft[flag,kruskal.stats<0.05],scale='c',RowSide=palette[as.numeric(yTrainAll[flag])+1])

################ Solution with all data

# Get MVN stats
rs.mvn = lapply(ixTrainAll, function(i){
  y = rs.fft[i,][flag[i],]
  S= cov(y)
  mu = apply(y,2,mean)
  list(mu=mu, Sigma=S)
})

# Compute Empirical Bayes posterior probabilities
rs.eb = t(apply(rs.fft[flag,],1,function(x)ebayes(x,rs.mvn)))
rs.eb = exp(sweep(rs.eb,1,apply(rs.eb,1,max),'-'))
rs.eb = (1/apply(rs.eb,1,sum))*rs.eb

# View clustering heatmap of posterior probabilities
heatmap(rs.eb, scale='n',RowSide=palette[as.numeric(yTrainAll[flag])+1],
  col=hsv(0.7,(0:255)/255,1), labRow=NA)

# Error rate
rs.xtab = table(apply(rs.eb,1,which.max),yTrainAll[flag])
sum(diag(rs.xtab))             # Total correctly classified
sum(diag(rs.xtab))/sum(flag) # Proportion correctly classified

################ Cross-validation

# Note:  it is not necessary to cross-validate the preprocessing

# Get CV folds
set.seed(10)
cv.folds = randomlyPartition(which(flag))
sapply(cv.folds,length)  # Number in each partition

# sample size in each fold
sapply(cv.folds, function(i)
  table(as.numeric(yTrainAll[setdiff(which(flag),i)]))
)

# Get cross-validated MVN stats
cv.mvn = list()
for(k in 1:10){
  cvi = setdiff(which(flag),cv.folds[[k]])
  cv.mvn[[k]] = lapply(ixTrainAll, function(i){
    i = intersect(i, cvi)
    y = rs.fft[i,][flag[i],]
    S= cov(y)
    mu = apply(y,2,mean)
    list(mu=mu, Sigma=S)
  })
}

# Use cross-validated MVN stats to get EB coefficients
cv.eb = matrix(NA, length(trainAll), length(ixTrainAll))
for(k in 1:10){
  tmp.eb = t(apply(rs.fft[cv.folds[[k]],],1,function(x)ebayes(x,rs.mvn)))
  tmp.eb = exp(sweep(tmp.eb,1,apply(tmp.eb,1,max),'-'))
  tmp.eb = (1/apply(tmp.eb,1,sum))*tmp.eb

  cv.eb[cv.folds[[k]],] = (1/apply(tmp.eb,1,sum))*tmp.eb
}

# View clustering heatmap of posterior probabilities
heatmap(cv.eb[flag,], scale='n',RowSide=palette[as.numeric(yTrainAll[flag])+1],
  col=hsv(0.7,(0:255)/255,1), labRow=NA)

# Error rate
cv.xtab = table(apply(cv.eb[flag,],1,which.max),yTrainAll[flag])
sum(diag(cv.xtab))             # Total correctly classified
sum(diag(cv.xtab))/sum(flag)   # Proportion correctly classified

################################################

################ Plots for Vignette

plotWeightedCenteredWaveForm = function(ts, env, errorClip=5000, legendpos='topright'){

   stat = getEnvelopePeakStats(ts, env, errorClip)

   xx = ts$data/stat["peakheight"]
   nx = length(xx)
   tt = ts$time-stat["peak"]

   sigma=stat["peakwidth"]
   sigma2=sigma*sigma

   wt = exp(-tt*tt/sigma2/2)

   plot(tt, ts$data, type='l', xlab='centered time', ylab='amplitude')
   wt = wt*(par()$usr[4]/max(wt))

   lines(tt, env, col='red', lwd=2)
   lines(tt, wt, col='royalblue', lty=2, lwd=2)
   lines(c(-1,1)*stat["peakwidth"], c(1,1)*par()$usr[3]/2, lwd=5, col='forestgreen')
   legend(legendpos, 
     c('abs val of waveform','GAM-based envelope', 'FFT weight', 'width parameter'), 
     lty=c(1,1,2,1), lwd=c(1,2,2,5), col=c('black','red','royalblue','forestgreen'), cex=0.8)
}

kruskal.stats.ord = order(kruskal.stats)
plotFFT.pal = rep(brewer.pal(7,'Set1'),2)
plotFFT.sym = c(rep(19,7),rep(17,6))
plotFFT = function(i,j){
  cl = as.numeric(yTrainAll[flag])+1
  plot(rs.fft[flag,i], rs.fft[flag,j],
     col=plotFFT.pal[cl], pch=plotFFT.sym[cl],
  xlab=paste('harmonic',colnames(rs.fft)[i]), 
  ylab=paste('harmonic',colnames(rs.fft)[j]))
}

plotMVN = function(i,j){
  mu = t(sapply(rs.mvn, function(u)u$mu[c(i,j)]))
  Sig = lapply(rs.mvn, function(u)u$Sigma[c(i,j),c(i,j)])
  L = lapply(Sig, function(s) t(chol(s)))

  tt = (1:360)*pi/180
  ccss = rbind(cos(tt),sin(tt))
  circs = list()
  for(k in 1:length(L)){
    circs[[k]] = sqrt(qchisq(.95,2))*sweep(L[[k]] %*% ccss,1,mu[k,],"+")
  }
  rgx = range(sapply(circs,function(u)u[1,]))
  rgy = range(sapply(circs,function(u)u[2,]))

  plot(rgx,rgy,type='n',  
    xlab=paste('harmonic',colnames(rs.fft)[i]), 
    ylab=paste('harmonic',colnames(rs.fft)[j]))
  for(k in 1:length(L)){
     lines(circs[[k]][1,], circs[[k]][2,],col=plotFFT.pal[k],lwd=2)
  }
}

plotWeightedCenteredWaveForm(tsAbs[[6]], rs.env[[6]], legendpos='topleft')
plotWeightedCenteredWaveForm(tsAbs[[5]], rs.env[[5]])
plotWeightedCenteredWaveForm(tsAbs[[106]], rs.env[[106]])
plotWeightedCenteredWaveForm(tsAbs[[3]], rs.env[[3]], legendpos='top')


plotFFT(kruskal.stats.ord[1],kruskal.stats.ord[2])
plotFFT(kruskal.stats.ord[1],kruskal.stats.ord[3])
plotFFT(kruskal.stats.ord[2],kruskal.stats.ord[3])
plotFFT(kruskal.stats.ord[2],kruskal.stats.ord[4])

plotMVN(kruskal.stats.ord[1],kruskal.stats.ord[2])
plotMVN(kruskal.stats.ord[1],kruskal.stats.ord[3])
plotMVN(kruskal.stats.ord[2],kruskal.stats.ord[3])
plotMVN(kruskal.stats.ord[1],kruskal.stats.ord[4])

# View clustering heatmap of coefficients
heatmap(pnorm(scale(rs.fft[flag,])),scale='c', col=cm.colors(255),
  RowSide=palette[as.numeric(yTrainAll[flag])+1], labRow=NA)

# View clustering heatmap of posterior probabilities
heatmap(cv.eb[flag,], scale='n',RowSide=palette[as.numeric(yTrainAll[flag])+1],
  col=hsv(0.7,(0:255)/255,1), labRow=NA)

isCorrectlyClassified = (apply(cv.eb[flag,],1,which.max)==as.numeric(yTrainAll[flag])+1)
entropy = -apply(cv.eb[flag,]*ifelse(cv.eb[flag,]==0,0,log(cv.eb[flag,])),1,sum)
boxplot(entropy~isCorrectlyClassified,xlab='Was Correctly Classified',ylab='Entropy')
wilcox.test(entropy~isCorrectlyClassified)
