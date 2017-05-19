# We estimate cell sizes via raking (iterative proportional fitting) with the following individual level variables and their population marginals:
# - sex (male, female)
# - age (Age 0 to 4, Age 5 to 7, Age 8 to 9, Age 10 to 14, Age 15, Age 16 to 17, Age 18 to 19, Age 20 to 24, Age 25 to 29, Age 30 to 44, Age 45 to 59, Age 60 to 64, Age 65 to 74, Age 75 to 84, Age 85 to 89, Age 90 and over)
## - education (No qualifications, Highest level of qualification: Level 1 qualifications, Highest level of qualification: Level 2 qualifications, Highest level of qualification: Apprenticeship, Highest level of qualification: Level 3 qualifications, Highest level of qualification: Level 4 qualifications and above, Highest level of qualification: Other qualifications, Schoolchildren and full-time students: Age 16 to 17, Schoolchildren and full-time students: Age 18 and over, Full-time students: Age 18 to 74: Economically active: In employment, Full-time students: Age 18 to 74: Economically active: Unemployed, Full-time students: Age 18 to 74: Economically inactive)
# education cannot be easily crossed with age (some are clearly non-combinable)
# - ethnicity (White, Mixed/multiple ethnic groups, Asian/Asian British, Black/African/Caribbean/Black British, Other ethnic group)
# - by region: North East, North West, Yorkshire and The Humber, East Midlands, West Midlands, East, London, South East, South West, Wales
library(foreign)
sex  <- read.csv("~/Desktop/PythonMLM/Election_Ex_Britain/2011_census/Sex_prep.csv")
sex <- sex[c("region", "Males", "Females")]
age  <- read.csv("~/Desktop/PythonMLM/Election_Ex_Britain/2011_census/Age_structure_prep.csv")
age[c("Age_0_17")] <- age[c("Age_0_4")] + age[c("Age_5_7")] + age[c("Age_8_9")] + age[c("Age_10_14")] + age[c("Age_15")] + age[c("Age_16_17")]
age[c("Age_85_over")] <- age[c("Age_85_89")] + age[c("Age_90_over")]
age <- age[c("Age_0_17","Age_18_19", "Age_20_24", "Age_25_29", "Age_30_44", "Age_45_59", "Age_60_64", "Age_65_74", "Age_75_84", "Age_85_over")]
#education  <- read.csv("~/Desktop/PythonMLM/Election_Ex_Britain/2011_census/Edu_prep.csv")
# "No_degree", "GCSE", "A-level" "Undergraduate". "Postgraduate"
ethnicity  <- read.csv("~/Desktop/PythonMLM/Election_Ex_Britain/2011_census/Ethnic_group_prep.csv")
region <- sex[1]


#this has issues as groups are not independent! we cannot
library(mipfp)
for (row in 1:dim(region)[1]){
	#education[row,14] <- sum(sex[row,2:3]) - sum(education[row,2:13])
	# prepare label
	name <- paste("demographics", region[row,1], sep = "_")
	# unlist variables for selected region
	sex_tmp <- sex[row,2:3]
	colnames(sex_tmp) <- NULL
	sex_tmp <- unlist(sex_tmp)
	age_tmp <- age[row,2:10]
	colnames(age_tmp) <- NULL
	age_tmp <- unlist(age_tmp)
	#education_tmp <- education[row,2:14]
	#colnames(education_tmp) <- NULL
	#education_tmp <- unlist(education_tmp)
	ethnicity_tmp <- ethnicity[row,2:6]
	colnames(ethnicity_tmp) <- NULL
	ethnicity_tmp <- unlist(ethnicity_tmp)
 	assign(name, list(sex_tmp, age_tmp, ethnicity_tmp))
 	#assign(name, list(sex_tmp, age_tmp, education_tmp, ethnicity_tmp))
 	#iterative proportional fitting
 	seed <-array(1,dim=c(2,9,5))
 	#seed <-array(1,dim=c(2,16,13,5))
 	target_data <- get(name)
 	target_list <- list( 1, 2, 3)# list( 1, 2, 3, 4)
 	name <- paste("joint_distribution", region[row,1], sep = "_")
 	ipfp <- Ipfp(seed, target_list, target_data)
 	assign(name, ipfp)
 	name <- paste("joint_distribution_values", region[row,1], sep = "_")
 	ipfp <- unlist(ipfp[1])
 	t <- NULL
 	for (row in 1:length(ipfp)){t[row] <- as.double(ipfp[row])}
 	assign(name, t)
}

label <- list()
c <- 1
for (i in 1:5){
	#for (j in 1:13){
	for (k in 1:16){
		for (l in 1:2){
			label[c] <- paste(l,k,i, sep = ".")
			c <- c + 1
		}
	}
	#}
}

total <- NULL
for (row in 1:dim(region)[1]){
	name <- paste("joint_distribution_values", region[row,1], sep = "_")
	data <- get(name)
	name <- paste("joint_distribution_pairs", region[row,1], sep = "_")
	assign(name, array(,dim=c(length(data),2)))
	final <- get(name)
	for (row_2 in 1:length(data)){
		final[row_2,1] = paste(label[[row_2]], row, sep=".")
		final[row_2,2] = data[row_2]
	}
	final <- as.data.frame(final)
	write.csv(final, paste("~/Desktop/PythonMLM/Election_Ex_Britain/2011_census/", name, ".csv", sep=""))
	assign(name, final)
	total <- rbind(total, final)
}
write.csv(total,"~/Desktop/PythonMLM/Election_Ex_Britain/2011_census/joint_distribution_pairs.csv")


#Later we can do more detailed estimates on constituency level by using the census data - constituency profile!




