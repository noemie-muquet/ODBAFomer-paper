library(moveHMM)

# =============================================================================
# input : file = csv file with all the gps data
# output : state probabilities associated with each time step of the input file
# =============================================================================

path <- getwd()
data_folder <- file.path(path, "data", "eval_data_timeseries_hmm")
file_path <- file.path(data_folder, "hmm_input.csv")
data<-read.table(file_path,sep=",",row.names=NULL,header=T)
names(data)
head(data)

# subset variables of interest
data <- data[, c(1, 3:5)]
names(data)<-c("ID", "lon","lat","odba_f")

head(data)

# first look at the data
plot(data$lon, data$lat)
plot(data$lon, data$lat,xlab="Longitude",ylab="Latitude")
plot(data$lon, data$lat,xlab="Longitude",ylab="Latitude",cex=.7,col="dark grey")
lines(data$lon, data$lat,col="black")

data.HMM<-prepData(data[,1:3],type="LL",coordNames=c("lon","lat"))
summary(data.HMM)
head(data.HMM)
plot(data.HMM,compact=T)

# HMM parameters setup
mu0 <- c(0.1, 0.01) 
sigma0 <- c(0.5, 0.01) 
zeromass0 <- c(0.01,0.01)
stepPar0 <- c(mu0,sigma0,zeromass0)
angleMean0 <- c(0, 0) 
kappa0 <- c(0.1,0.5) 
anglePar0 <- c(angleMean0,kappa0)

# call to fitting function
hmm <- fitHMM(data=data.HMM,nbStates=2,stepPar0=stepPar0, anglePar0=anglePar0)
hmm
CI(hmm)
plot(hmm)

# state decoding
states <- viterbi(hmm)
states[1:50]

# state probability
sp <- stateProbs(hmm)
head(sp)
plotStates(hmm)

results_path = file.path(data_folder, "states_probabilities.csv")

# save states
write.table(sp, results_path, sep = ",", row.names = FALSE, col.names =  c("p1", "p2"))
