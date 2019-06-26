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

# Density of multivariate normal (MVN) distribution
dmvnorm = function(x, mu, Sigma){
   r = x-mu
   exp(-(log(2*pi)*length(mu)+det(Sigma,log=TRUE)+r%*%solve(Sigma,r))/2)
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

# Clip data from front and end of .wav data
clipTransformWave = function(wv, transform=function(x)x, clip1=500, clip2=500){
  xx = as.vector(wv$sound)
  nx = length(xx)
  xx = xx[-c(1:clip1,nx-(1:clip2))]
  transform(xx)
}

# Plot raw .wav data (clipped)
plotRawWave = function(wv, transform=function(x)x, clip1=500, clip2=500,...){
  xx = clipTransformWave(wv, transform, clip1, clip2)
  nx = length(xx)
  tt = 1:nx
  plot(tt, transform(xx), type='l', xlab='time', ylab='amplitude',...)
}

# Use generalized additive model (GAM) to calculate a smooth envelope
getSmoothEnvelope = function(wv, f=NULL, clip1=500, clip2=500){
  y = clipTransformWave(wv, transform=abs)
  x = 1:length(y)
  gf = gam(y~s(x))
  predict(gf)
}

# Get peak stats from GAM-based envelope
getEnvelopePeakStats = function(pr, errorClip=5000){
  xx = as.numeric(names(pr))

  x0 = which.max(pr) # Get time corresponding to maximum envelope

  # If maximum occurs at the beginning, clip some more from the beginning
  if(x0==1) x0=which.max(pr[-(1:errorClip)])+errorClip
  y0 = pr[x0]
  d1 = diff(pr)  # First difference
  d2 = diff(d1)  # Second difference
  a0 = d2[x0-1]
  
  # Half-width calculated as the difference between peak and x-intercepts
  # of the parabola implied by the 2nd difference and the peak height
  halfwidth = sqrt(-y0/a0) 

  c(`peakheight`=y0,`peak`=xx[x0],`curvature`=a0,`peakwidth`=halfwidth)
}

# Weighted Fast-Fourier-Transform (general function)
FFT = function(t,x,K=10,L=length(t),weight=rep(1,length(t)),saveBasis=FALSE){
  tt = 2*pi*t/L
  ss = sin(outer(tt, 1:K, '*'))[,K:1]
  cc = cos(outer(tt, 0:K, '*'))
  B = cbind(ss,cc)
  colnames(B) = c(-(K:1),0,1:K)
  Bt = t(weight*B)
  out = list(coef = solve(Bt%*%B, Bt%*%as.vector(x)))
  rownames(out$coef) = colnames(B)

  if(saveBasis) out$basis = B
  out
}

# Weighted FFT for this specific problem (inputs: wave, center, width, etc.)
getWeightedFFT = function(wv, mu, sigma, scale, sigmaMult=3, K=10, clip1=500, clip2=500,...){
  #xx = as.vector(wv$sound)
  xx = abs(as.vector(wv$sound))/scale
  nx = length(xx)
  xx = xx[-c(1:clip1,nx-(1:clip2))]

  nx = length(xx)
  tt = 1:nx-mu

  FFT(tt,xx,K=K,L=sigmaMult*sigma,weight=dnorm(tt,0,sigma))
}

# Create a randomly generated partion for K-fold cross-validation
randomlyPartition = function(i, folds=10){
  split(i, sample(1:folds, length(i), replace=TRUE))
}

################ Read data

train1 = readWaveFiles()
yTrain1 = substring(names(train1),1,2)
ixTrain1 = split(1:length(yTrain1), yTrain1)

train2 = readWaveFiles('moreNumbers\\numbers')
yTrain2 = substring(names(train2),1,2)
ixTrain2 = split(1:length(yTrain2), yTrain2)

trainAll = c(train1, train2)
yTrainAll = c(yTrain1, yTrain2)
ixTrainAll = split(1:length(yTrainAll), yTrainAll)

# Paltette for classes
palette = c(brewer.pal(7,'Set1'), brewer.pal(6,'Set3'))

################ Preprocessing

# First time:  get smooth envelopes and save (due to length of time to process)
#rs.env = lapply(trainAll, getSmoothEnvelope)
#saveRDS(rs.env, file='gam-envelopes-190624.rds')

# Later times (after smooth envelopes have been saved)
rs.env = readRDS('gam-envelopes-190624.rds')

# Get the peak envelope stats for all files
rs.stat = t(sapply(rs.env,getEnvelopePeakStats))

#remove garbage from column names
colnames(rs.stat) = gsub('[.][[:digit:]]*$','', colnames(rs.stat)) 

# Create a matrix of FFT coefficients
rs.fftList = mapply(getWeightedFFT, wv=trainAll, 
  mu=rs.stat[,"peak"], sigma=rs.stat[,"peakwidth"], scale=rs.stat[,"peakheight"], 
  sigmaMult=5, K=15, SIMPLIFY=FALSE)
rs.fft = t(sapply(rs.fftList, function(u)u$coef))
colnames(rs.fft) = rownames(rs.fftList[[1]]$coef)

# filter errors
flag = !apply(is.na(rs.fft),1,any) 
sum(!flag)   # How many errors?

# Kruskal-Wallis statistics for each coefficient (across all classes)
kruskal.stats = apply(rs.fft[flag,],2,function(u)kruskal.test(u~factor(yTrainAll[flag]))$p.value)
sort(kruskal.stats)

# View clustering heatmap of coefficients
heatmap(rs.fft[flag,kruskal.stats<0.05],scale='c',RowSide=palette[as.numeric(yTrainAll[flag])+1])

################ Solution with all data

# Select features (turns out using all of them provides best fit)
rs.selFeatures = which(kruskal.stats<=1) 

# Get MVN stats
rs.mvn = lapply(ixTrainAll, function(i){
  y = rs.fft[i,][flag[i],rs.selFeatures]
  S= cov(y)
  mu = apply(y,2,mean)
  list(mu=mu, Sigma=S)
})

# Compute Empirical Bayes posterior probabilities
rs.eb = t(apply(rs.fft[flag,rs.selFeatures],1,function(x)ebayes(x,rs.mvn)))
rs.eb = (1/apply(rs.eb,1,sum))*rs.eb

# View clustering heatmap of posterior probabilities
heatmap(rs.eb, scale='n',RowSide=palette[as.numeric(yTrainAll[flag])+1],
  col=hsv(0.7,(0:255)/255,1), labRow=NA)

# Error rate
rs.xtab = table(apply(rs.eb,1,which.max),yTrainAll[flag])
sum(diag(rs.xtab))             # Total correctly classified
sum(diag(rs.xtab))/sum(flag)   # Proportion correctly classified

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
    y = rs.fft[i,][flag[i],rs.selFeatures]
    S= cov(y)
    mu = apply(y,2,mean)
    list(mu=mu, Sigma=S)
  })
}

# Use cross-validated MVN stats to get EB coefficients
cv.eb = matrix(NA, length(trainAll), length(ixTrainAll))
for(k in 1:10){
  tmp.eb = t(apply(rs.fft[cv.folds[[k]],],1,function(x)ebayes(x,rs.mvn)))
  cv.eb[cv.folds[[k]],] = (1/apply(tmp.eb,1,sum))*tmp.eb
}

# View clustering heatmap of posterior probabilities
heatmap(cv.eb[flag,], scale='n',RowSide=palette[as.numeric(yTrainAll[flag])+1],
  col=hsv(0.7,(0:255)/255,1), labRow=NA)

# Error rate
cv.xtab = table(apply(cv.eb[flag,],1,which.max),yTrainAll[flag])
sum(diag(cv.xtab))             # Total correctly classified
sum(diag(cv.xtab))/sum(flag)   # Proportion correctly classified

