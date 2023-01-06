# clears workspace:  
rm(list=ls(all=TRUE)) 

# Set working directory  (replace with your own path)
#dir <- "your/path/here/RLVOLUNP_CIT_ana/"
# Hint: 
# On Mac/Linux: open a terminal session in the folder containing the analyses and type 'pwd' without quotes
# On Windows: open a command prompt in the folder containing the analyses and type 'echo %cd%' without quotes 

dir <- "/your/path/here/RLVOLUNP_CIT_ana/"

setwd(dir)

# Load and if necessary install the required libraries
library(R2WinBUGS)  # Only necessary for posterior plots for JZS prior
library(MCMCpack)
library("hypergeo")

# Load the functions necessary to compute the replication prior (file saved in working directory)
source(paste(dir,'/toolbox/Repfunctionspack6.R', sep = "")) 
source(paste(dir,'/toolbox/ReplicationFunctionsCorrelation.R', sep = ""))

N1 = 137;
N2 = 123;
###############################################################################
# Figure 4 
# replication of accuracy ~ CIT (regression) Figure 4b
BFSALL(-2.5694, -2.2878, N1, N2, N1, N2, Type = "REPBF")

# replication of accuracy ~ ICAR (regression) Figure 4b
BFSALL(2.3381, 2.6697, N1, N2, N1, N2, Type = "REPBF")

# replication of switch rate ~ CIT (regression) Figure 4b
BFSALL(2.3806, 2.1493, N1, N2, N1, N2, Type = "REPBF")

# replication of switch rate ~ ICAR (regression) Figure 4b
BFSALL(-2.7796, -1.6883, N1, N2, N1, N2, Type = "REPBF")

###############################################################################
# Figure 5
# replication of choice temperature ~ CIT (regression) Figure 5b
BFSALL(3.0186, 1.8096, N1, N2, N1, N2, Type = "REPBF")

# replication of choice temperature ~ ICAR (regression) Figure 5b
BFSALL(-2.3676, -1.7164, N1, N2, N1, N2, Type = "REPBF")

###############################################################################
# Supplementary Figure 5b
# replication of accuracy ~ obs comp/ocir (regression) Figure 6b
BFSALL(-2.7914, -1.2339, N1, N2, N1, N2, Type = "REPBF")

# replication of switch rate ~ obs comp/ocir (regression) Figure 6b
BFSALL(2.5689, 1.2412, N1, N2, N1, N2, Type = "REPBF")

# replication of choice temp ~ obs comp/ocir (regression) Figure 6d
BFSALL(3.3143, 1.5139, N1, N2, N1, N2, Type = "REPBF")


