################################
# LocalScore.R  #
################################
#
#
# The following functions are implemented:
#
#	. rankNormalization(x)
#
#	. LocalSimilarity(x, y, maxDelay=3, rankscale=FALSE)
#
# Last updated: Jan 27, 2017, Fang Zhang.  @All rights reserved.
#############################################################################
################################
# rankNormalization(x):
#
#	This function performs rank normalization of given vector
#
###############################
rankNormalization <- function(x)
{
  xLength <- length(x)
  rankScore <- rep(0,xLength)
  rankScore <- scale(qnorm(rank(x)/(xLength+1)))
  return(rankScore)
}

##################################################################
# LocalSimilarity<- function(x, y, maxDelay=3, rankscale=TRUE)
#
#  This function computes the local similarity score for two sequences.
#
# INPUT:
# ======
#
#	x, y	: sequences to copute LS score
#	maxDelay	: maximum time shift allowed in computing LS score.
#	rankscale		: If TRUE, perform rankNormalization first; False, otherwise.
#
# RETURN:
# ======
#
#  A six element vector contains: c(scoreMax, startX, startY, delay, length, PosOrNeg)
#

LocalSimilarity <- function(x, y, maxDelay=3, rankScale = TRUE){
  
  if (rankScale == TRUE)
  {
    x <- rankNormalization(x)
    #print(x)
    y <- rankNormalization(y)
    #print(y)
  }
  
  scoreMax <- 0;
  PosOrNeg <- 0;
  startX <- 0;
  startY <- 0;
  numTimepoints <- length(x)
  posMatrix <- matrix(0, numTimepoints+1, numTimepoints+1)
  negMatrix <- matrix(0, numTimepoints+1, numTimepoints+1)
  for (i in 1:numTimepoints){
    for (j in max(1, i-maxDelay):min(i+maxDelay, numTimepoints)){
      posMatrix[i+1,j+1] <- max(0, posMatrix[i,j] + x[i] * y[j])
      negMatrix[i+1,j+1] <- max(0, negMatrix[i,j] - x[i] * y[j])
    }
  }
  posMatrix <- posMatrix[-1,]
  posMatrix <- posMatrix[,-1]
  negMatrix <- negMatrix[-1,]
  negMatrix <- negMatrix[,-1]
  scorePosmax <- max(posMatrix)
  scoreNegmax <- max(negMatrix)
  scoreMax <- max(scorePosmax, scoreNegmax)
  if( scorePosmax > scoreNegmax) PosOrNeg = 1 else PosOrNeg = -1
  if (scorePosmax > scoreNegmax) {
    Maxposition <- which(posMatrix == scorePosmax, arr.ind = TRUE)
  }else  Maxposition <- which(negMatrix == scoreNegmax, arr.ind = TRUE)
  delay <- Maxposition[1] - Maxposition[2]
  subinternal<-NULL
  for(i in max(1, delay):min(numTimepoints,numTimepoints + delay)){
    if (scorePosmax > scoreNegmax) {
      subinternal <- c(subinternal,posMatrix[i, i-delay])
    }else  subinternal <- c(subinternal,negMatrix[i,i-delay])
  }
  if (delay>0) {
    startX <- max(1,which(subinternal[1:Maxposition[1]] == 0) + 1) + delay
  }else{
    startX <- max(1,which(subinternal[1:Maxposition[1]] == 0) + 1)
  }
  startY <- startX - delay
  lengths <- Maxposition[1] - startX + 1
  value <- t(c(scoreMax, startX, startY, delay, lengths, PosOrNeg))
  colnames(value)<-c('Maxscore', 'startX', 'startY', 'delay', 'length', 'PosOrNeg')
  return(value)
}

######################
# permutationTest(x, y, numPermu=1000, maxDelay=3,scale=TRUE)
#
# This function computes the significance level of the LS score by permutation test
#
######################
permutationTest <- function(x, y, numPermu=1000, maxDelay=3,scale=TRUE){
  scoreArray <- rep(0.0, numPermu+1);
  
  if (scale == TRUE){
    x <- rankNormalization(x)
    y <- rankNormalization(y)
  }
  numTimePoints <- length(x)
  scoreMax1 <- LocalSimilarity(x, y, maxDelay)[1];
  scoreArray[1] <- scoreMax1;
  highScoreCnt <- 0;
  
  for(idx in 1:numPermu)
  {
    dtt1 <- x[sample(numTimePoints)]
    scoreTmp <- LocalSimilarity(dtt1, y, maxDelay)[1];
    
    scoreArray[idx+1] <- scoreTmp;
    highScoreCnt <- highScoreCnt + (scoreTmp >= scoreMax1);
  }
  
  pValue <- 1.0 * highScoreCnt / numPermu;
  
  return(pValue);
}
kolmogorovSmirnovTestWithDirection <- function(x, y){
  ks_result <- ks.test(x, y)
  ks_statistic <- ks_result$statistic  
  ks_p_value <- ks_result$p.value      

  x_cdf <- ecdf(x)  
  y_cdf <- ecdf(y) 
  

  x_leads_y <- max(x_cdf(x)) > max(y_cdf(y)) 
  y_leads_x <- !x_leads_y 
  
  
  direction <- ifelse(x_leads_y, "x leads y", "y leads x")
  
  return(c(ks_statistic, ks_p_value, direction))
}

data = read.csv("infant_lsa.csv")

results <- data.frame(Species1 = character(0), Species2 = character(0), SimilarityScore = numeric(0), pValue = numeric(0))
for (i in 1:(nrow(data)-1)) {
  for (j in (i+1):nrow(data)) {
    species1 <- rownames(data)[i]
    species2 <- rownames(data)[j]
  
    x <- as.numeric(data[i, ])
    y <- as.numeric(data[j, ])
 
    similarity <- LocalSimilarity(x, y, maxDelay=3)
    Maxscore <- similarity[1, "Maxscore"]
    PosOrNeg <- similarity[1, "PosOrNeg"]
    
    ks_result <- kolmogorovSmirnovTestWithDirection(x, y)
    ks_statistic <- ks_result[1]
    ks_p_value <- ks_result[2]
    direction <- ks_result[3]

    similarity_score <- Maxscore * PosOrNeg
    
    p_value <- permutationTest(x, y)

    results <- rbind(results, data.frame(Species1 = species1, 
                                         Species2 = species2, 
                                         SimilarityScore = similarity_score, 
                                         pValue = p_value,
                                         KSStatistic = ks_statistic,
                                         KSPValue = ks_p_value,
                                         Direction = direction))
  }
}


filtered_results <- subset(results, pValue < 0.05)
