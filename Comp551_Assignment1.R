#IMPORTANT!!:
# You need to exclude year 2013
# You need to exclude everyone who participated only once

#To set working directory
setwd("R")

#Tool: To see working directory
getwd()
full_data <- read.csv("modified_file_3.csv")

full_data$Time_hours <- full_data2$Time_seconds/3600
full_data <- subset(full_data, Year != 2013)

#Extract the file into a dataframe




ratio = mean(subset(full_data2$Time_hours, full_data2$Year != 2013))/mean(subset(full_data2$Time_hours, full_data2$Year == 2013))

for(i in 1:nrow(full_data2)){
  
  if (full_data2[i,]$Year == 2013)
   full_data2[i,]$Time_hours <- full_data2[i,]$Time_hours*ratio
}


# Sampling the data into a random order
full_data <- full_data[sample(nrow(full_data)),]

full_data <- subset(full_data, average < 15443.4 | average > 15443.6)

#Creating a new column for the time in hours

#Tool: How to "see" the dataframe
str(full_data)


#Separating the full data set into one for training & Validation (data) and one for testing (test)
training <- full_data[1:(nrow(full_data)*0.8),]
validation <- full_data[((nrow(full_data)*0.8)+1):(nrow(full_data)),]


#How to fit a model to the data using least-squares
#This is a list of models; to create a new model, just change the number inside model[[x]]
model <- NULL
model[[1]] <- lm(Time_hours ~ average, training)
model[[2]] <- lm(Time_hours ~ average + Age.Category, training)
model[[3]] <- lm(Time_hours ~ average + Age.Category + Year, training)
model[[4]] <- lm(Time_hours ~ average + Age.Category + Year + Sex, training)
model[[5]] <- lm(Time_hours ~ average + Age.Category + Year + Sex + participations, training)
model[[6]] <- lm(Time_hours ~ average + Age.Category + Year + Sex + participations +poly(average, 2), training)
model[[7]] <- lm(Time_hours ~ average + Age.Category + Year + Sex + participations + poly(average, 2) + poly(Age.Category,2) + poly(average,3) + average*Age.Category, training)

#model 10 is good

residuals <- NULL
for (x in 1:length(model)){
  if(!is.null(model[[x]])){
    a <- anova(model[[x]])
residuals <- rbind(residuals, c(a[nrow(a),2]/nrow(training), sum((predict(model[[x]], newdata = validation) - validation$Time_hours)^2)/nrow(validation)))
}}
colnames(residuals) <- c("Training error", "Validation error")
