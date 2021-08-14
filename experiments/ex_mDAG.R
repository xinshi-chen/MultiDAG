rm(list=ls())
# read source files
source("functions.R")
library(reticulate)
np <- import("numpy")

########################################################################
# EXAMPLE
########################################################################
##First read in the arguments listed at the command line
args=(commandArgs(TRUE))

##args is now a list of character vectors
## First check to see if arguments are passed.
## Then cycle through each element of the list and evaluate the expressions.
if(length(args)==0){
  print("No arguments supplied.")
  ##supply default values
  p = 32
  n_sample = 10
  K = 256
  group_size = 4
  s = 120
  s0 = 40
  d = 5
}else{
  for(i in 1:length(args)){
    eval(parse(text=args[[i]]))
  }
}

dir = paste("../data/p-", p, "_n-", n_sample, "_K-", K, "_s-", s, "_s0-", s0, "_d-", d, "_w_range_l-0.5_w_range_u-2.0", sep="")
real_A <- np$load(paste(dir, "/G.npy", sep=""))
real_X <- np$load(paste(dir, "/X.npy", sep=""))

alpha = 0.999
gamma = 1
nu0 = 0.1
c1 = 1 #c1 in ESC paper, doesn't matter when set to 1
c2 = 2
b = 1/(p*(K - 1)) #c2 in the paper
niter = 1000
nburn = 0.2*niter
nadap = 0

result <- array(0, dim=c(K, 5))
N = K / group_size
time_cost <- array(0, dim=c(N,1))
########################################################################
# joint DAG posterior inference with ESC prior
########################################################################
for (idx in 1:N){
  A = real_A[(group_size*(idx-1) + 1):(group_size*idx),,]
  X = real_X[(group_size*(idx-1) + 1):(group_size*idx),,]
  time = Sys.time()
  library(parallel)
  source("functions.R")
  res = mdag(X, alpha, gamma, nu0, c1, c2, c3=NULL, b, niter, nburn)
  elapsed = Sys.time() - time
  time_cost[idx,1] = elapsed
  
  #sort return list
  l1 = list()
  for (k in 1:group_size) {
    l2 = list(NULL)
    for (j in 1:(p-1)) {
      m <- matrix(0, niter, j)
      for (t in (nburn + 1):(nburn + niter)) {
        m[t-nburn, ] = res[[j]][[t]][k, ]
      }
      l2[[j]] = m
    }
    l1[[k]] = l2
  }
  
  ########################################################################
  # results
  ########################################################################
  # res FDR, TPR, FPR, SHD, NNZ
  for (k in 1:group_size){
    incl <- matrix(0, p, p)
    for (j in 2:p){
      Sj.mat = l1[[k]][[j-1]]
      incl[j, seq(1:(j-1))] <- apply(Sj.mat, 2, mean)
    }
    result[group_size*(idx-1) + k,] <- evaluation.dag((A[k,,][lower.tri(A[k,,])] != 0),1*(incl[lower.tri(incl)]>0.5))
  }
}
apply(time_cost, 2, mean)
apply(time_cost, 2, sd)
apply(result, 2, mean)
apply(result, 2, sd)