setwd("R")

#Tool: To see working directory
getwd()

#Extract the file into a dataframe
data <- read.csv("modified_file_3.csv")

ratio = mean(subset(full_data$Time_hours, full_data$Year != 2013))/mean(subset(full_data$Time_hours, full_data$Year == 2013))

for(i in 1:nrow(full_data)){
  
  if (full_data[i,]$Year == 2013)
    full_data[i,]$Time_hours <- full_data[i,]$Time_hours*ratio
}


# Sampling the data into a random order
full_data <- full_data[sample(nrow(full_data)),]

#Creating a new column for the time in hours
full_data$Time_hours <- full_data$Time_seconds/3600

linreg <- function(data, predictors){
y <- full_data$Time_hours
x <- as.matrix(cbind(full_data[, predictors], rep(1, nrow(full_data))))
tx <- t(x)
tx_x <- tx%*%x
inv <- solve(tx_x)
w1 <- inv%*%tx
w <- w1%*%y
return(w)
}

linreg(full_data, c(1,3,8))

logreg <- function(extended_data, predictors){

x <- as.matrix(cbind(full_data[, predictors], rep(1, nrow(full_data))))
w <- c(10, rep(-5, length(predictors)))
y <- full_data$X2016

for(c in 1:10000){
tw <- t(w)
sum <- 0
sum_error = 0 
alpha <- 1/nrow(full_data)

for (i in 1:nrow(full_data)){
  a <- as.integer(tw%*%x[i,])
  sigma <- 1/(1 + exp(-a))
  error = y[i] - sigma
  sum_error = sum_error + error
  sum <- x[i,]*error + sum
}

w = w + alpha*sum
print(w)
print(sum_error)
print(sum)
past_error = sum_error
}
return(w)
}

model <- glm(X2016 ~ X2015 + X2014 + X2013 + X2015*X2014 + Year + Sex + Age.Category,family=binomial(link='logit'),data=data)

Coefficients:
  (Intercept)         Year        X2015  
-8.0229       8.1241       0.2147  
