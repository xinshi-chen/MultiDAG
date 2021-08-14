mdag <- function(X, alpha, gamma, nu0, c1, c2, c3=NULL, b, niter, nburn) {
  K = dim(X)[1]
  n = dim(X)[2]
  p = dim(X)[3]
  # initialize
  S = array(0, dim=c(K, p, p))
  S_chain = list(NULL)
  
  update_row = function(j) {
    # set Rj value
    if(is.null(c3)){
      Rj = floor( n/(log(p, base=10)*log(n, base=10)) )
    }else{
      Rj = floor( n/log(p)*min(c3, 1/log(n)) )
    }
    Sj_chain = list(matrix(S[, j, 1:(j-1)], nrow = K))
    for (t in 2:(niter + nburn)) {
      Sj_curr = Sj_chain[[t-1]]
      for (k in 1:K) {
        Skj_curr = Sj_curr[k, ]
        Skj_new = Sprop(Skj_curr, j, Rj)
        Sj_new = Sj_curr
        Sj_new[k, ] = Skj_new
        log_ratio = logpost(X[k,,], j, Rj, Skj_new, alpha, gamma, nu0, c1, c2) - logpost(X[k,,], j, Rj, Skj_curr, alpha, gamma, nu0, c1, c2) + lkernelprob(Skj_curr, Skj_new, j) + b * (sum((colSums(Sj_new)) ^ 2) - sum(colSums(Sj_new ^ 2)) - (sum((colSums(Sj_curr)) ^ 2) - sum(colSums(Sj_curr ^ 2))))
        prob = min(1, exp(log_ratio))
        if (rbinom(1, 1, prob) == 1) {
          Sj_curr[k, ] = Skj_new
        }
      }
      Sj_chain[[t]] = Sj_curr
    }
    #cat("Posterior sampling for ", j,"th row is completed. . . . . .\n")
    return(Sj_chain)
  }
  #for parallel computing
  mclapply(2:p, update_row, mc.cores = 1)
}

Sprop <- function(S, j, Rj){
  s = sum(S)
  upper.ind = min(j-1, Rj)
  
  if(s == 0){ # if current S has no index
    S[sample(which(S == 0), 1)] = 1
  }else if(s == upper.ind){ # if current S has maximum index (upper.ind)
    S[sample(which(S > 0), 1)] = 0
  }else{
    
    if(runif(1) <= 0.5){ # introducing additional one 0 to current S
      if(s == 1){
        S[which(S == 1)] = 0
      }else{
        S[sample(which(S > 0), 1)] = 0
      }
      
    }else{ # introducing additional one 1 to current S
      if(j-1-s == 1){
        S[which(S == 0)] = 1
      }else{
        S[sample(which(S == 0), 1)] = 1
      }
    }
    
  }
  return(S)
}

logpost <- function(X, j, Rj, S, alpha, gamma, nu0, c1, c2){
  n = nrow(X)
  p = ncol(X)
  s = sum(S)
  
  dS.hat = dShat(X, j, S)
  
  logpij = dpois(s, lambda = 1, log = T) - lchoose(j-1, s) + log(s <= min(j-1, Rj)) # Poisson prior for the model size
  
  logpost.val = logpij - s*log(1 + alpha/gamma)/2 - (alpha*n + nu0)*log(dS.hat)/2
  
  return(logpost.val)
}


dShat <- function(X, j, S){
  n = nrow(X)
  s = sum(S)
  tilde.Xj = as.matrix(X[, j])
  
  if(s == 0){
    return( sum((tilde.Xj)^2)/n )
  }else{
    Zj = as.matrix(X[, 1:(j-1)])
    X.Sj = as.matrix(Zj[, S>0])
    VhatX = sum(tilde.Xj^2)/n   # scalar
    Covhat = t(X.Sj)%*%matrix(tilde.Xj)/n   # s X 1 matrix
    res = VhatX - t(Covhat)%*%solve( t(X.Sj)%*%X.Sj/n )%*%Covhat
    return( res )
  }
}


lkernelprob <- function(Sj, Sj.new, j){
  if(sum(Sj - Sj.new) == -1){
    return(log(j-1 - sum(Sj)) - log(sum(Sj.new)))
  }else{
    return(log(sum(Sj)) - log(j-1 - sum(Sj.new)))
  }
}



evaluation.dag <- function(Adj1, Adj2){
  true.index <- which(Adj1==1)
  false.index <- which(Adj1==0)
  positive.index <- which(Adj2==1)
  negative.index <- which(Adj2==0)
  
  TP <- length(intersect(true.index,positive.index))
  FP <- length(intersect(false.index,positive.index))
  FN <- length(intersect(true.index,negative.index))
  TN <- length(intersect(false.index,negative.index))
  
  MCC.denom <- sqrt(TP+FP)*sqrt(TP+FN)*sqrt(TN+FP)*sqrt(TN+FN)
  if(MCC.denom==0) MCC.denom <- 1
  MCC <- (TP*TN-FP*FN)/MCC.denom
  if((TN+FP)==0) MCC <- 1
  
  Precision <- TP/(TP+FP)
  if((TP+FP)==0) Precision <- 1
  FDR <- FP/(TP+FP)
  if((TP+FP)==0) FDR <- 1
  Recall <- TP/(TP+FN)
  if((TP+FN)==0) Recall <- 1
  Sensitivity <- Recall
  Specific <- TN/(TN+FP)
  if((TN+FP)==0) Specific <- 1
  TPR <- Sensitivity
  FPR <- 1 - Specific
  SHD <- FP + FN
  NNZ <- TP + FP
  return(c(FDR, TPR, FPR, SHD, NNZ))
  # return(list(FDR=FDR,TPR=Sensitivity,FPR=1-Specific, SHD=FP+FN, NNZ=TP+FP))
  # return(list(FDR=FDR,TPR=Sensitivity,FPR=1-Specific,MCC=MCC,error = FP+FN, FP=FP,TP=TP,FN=FN,TN=TN))
}


evaluation.vec <- function(beta1, beta2){
  true.index <- which(beta1==1)
  false.index <- which(beta1==0)
  negative.index <- which(beta2==0)
  
  TP <- length(intersect(true.index,positive.index))
  FP <- length(intersect(false.index,positive.index))
  FN <- length(intersect(true.index,negative.index))
  TN <- length(intersect(false.index,negative.index))
  MCC.denom <- sqrt(TP+FP)*sqrt(TP+FN)*sqrt(TN+FP)*sqrt(TN+FN)
  if(MCC.denom==0) MCC.denom <- 1
  MCC <- (TP*TN-FP*FN)/MCC.denom
  if((TN+FP)==0) MCC <- 1
  
  Precision <- TP/(TP+FP)
  if((TP+FP)==0) Precision <- 1
  FDR <- FP/(TP+FP)
  if((TP+FP)==0) FDR <- 0
  Recall <- TP/(TP+FN)
  if((TP+FN)==0) Recall <- 1
  Sensitivity <- Recall
  Specific <- TN/(TN+FP)
  if((TN+FP)==0) Specific <- 1
  return(list(FDR=FDR,TPR=Sensitivity,FPR=1-Specific,MCC=MCC,error = FP+FN, FP=FP,TP=TP,FN=FN,TN=TN))
}


DAGLassoseq <- function(Y, lambda.seq, maxitr=100, tol=1e-4){
  require(lassoshooting)  
  p = ncol(Y)
  n = nrow(Y)
  S = (t(Y) %*% Y)/n
  T = diag(p)
  D = rep(1, p)
  itr_log = eps_log = NULL
  for (k in 2:p){
    nuk_old = nuk_new = c(rep(0, k-1), 1)
    r = 0
    km1_ = 1:(k-1)
    repeat {
      r = r + 1
      nuk_new[k] = D[k]
      output = lassoshooting(XtX= S[km1_,km1_,drop=FALSE],Xty=-S[km1_,k], maxit = 100, lambda=0.5*nuk_new[k]*lambda.seq[k])
      nuk_new[km1_] = output$coefficients
      maxdiff = max(abs(nuk_new - nuk_old))
      if (maxdiff < tol || r >= maxitr){
        T[k, km1_] = nuk_new[km1_]
        eps_log = c(eps_log, maxdiff)
        itr_log = c(itr_log, r)
        break
      } else {
        nuk_old = nuk_new
      }
    }
  }
  Adj <- matrix(0,p,p)
  Adj[which(T!=0)] <- 1
  for(i in 1:p) Adj[i,i] = 0
  return(Adj)
}

Joint_auc <- function(gamma0, inc.prob){
  
  thres = inc.prob[order(inc.prob)] # thresholds for ROC curve
  thres[-1] = thres[-1] + abs(min(diff(thres)))/2
  thres[1] = thres[1] - 0.001
  
  TPR = rep(0, length(thres))
  FPR = rep(0, length(thres))
  for(i in 1:length(thres)){
    gamma.est = (inc.prob > thres[i])
    TPR[i] = sum(gamma0*gamma.est)/sum(gamma0)
    FPR[i] = sum(gamma.est-gamma0 == 1)/sum(1-gamma0)
  }
  
  TPR = TPR[order(TPR)]
  FPR = FPR[order(FPR)]
  
  # inputs already sorted, best scores first 
  dFPR <- c(diff(FPR), 0)
  dTPR <- c(diff(TPR), 0)
  sum(TPR * dFPR) + sum(dTPR * dFPR)/2
}