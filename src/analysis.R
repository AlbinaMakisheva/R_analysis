library(tidyverse)
library(caret)
library(reshape2)
library(randomForest)

# Download and unzip UCI Student Performance dataset
dataset_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
zip_file <- "student.zip"
csv_file <- "student-mat.csv"

if (!file.exists(zip_file)) {
  download.file(dataset_url, zip_file, method = "auto")
  print("Dataset downloaded successfully!")
}

if (!file.exists(csv_file)) {
  unzip(zip_file)
  print("Dataset unzipped successfully!")
}

df <- read.csv(csv_file, sep = ";")

# Part 1: Dataset Preparation
print("First few rows of the dataset:")
print(head(df))

print("Dataset structure:")
print(str(df))

print("Checking for missing values...")
missing_values <- sum(is.na(df))
print(paste("Missing values:", missing_values))

print("Checking for duplicates...")
duplicates <- sum(duplicated(df))
print(paste("Duplicate rows:", duplicates))

# Remove duplicates
if (duplicates > 0) {
  df <- df[!duplicated(df), ]
  print("Duplicates removed.")
}

# Convert categorical columns to factors
df <- df %>%
  mutate(
    sex = as.factor(sex),
    address = as.factor(address),
    famsize = as.factor(famsize),
    Pstatus = as.factor(Pstatus),
    Medu = as.factor(Medu),
    Fedu = as.factor(Fedu),
    Mjob = as.factor(Mjob),
    Fjob = as.factor(Fjob),
    reason = as.factor(reason),
    guardian = as.factor(guardian),
    schoolsup = as.factor(schoolsup),
    famsup = as.factor(famsup),
    paid = as.factor(paid),
    activities = as.factor(activities),
    nursery = as.factor(nursery),
    higher = as.factor(higher),
    internet = as.factor(internet),
    romantic = as.factor(romantic),
    school = as.factor(school)
  )

# Part 2: Exploratory Data Analysis
print("Summary statistics of the dataset:")
print(summary(df))

# Visualize score distributions
ggplot(df, aes(x = G3)) + 
  geom_histogram(fill = "blue", bins = 30, alpha = 0.7) + 
  ggtitle("Final Grade (G3) Distribution") + 
  theme_minimal()

# Visualize correlations between features
correlation_matrix <- df %>%
  select(G1, G2, G3) %>%
  cor()

print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlations using heatmap
ggplot(melt(correlation_matrix), aes(Var1, Var2, fill = value)) + 
  geom_tile() + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() + 
  ggtitle("Correlation Heatmap")

# Part 3: Feature Selection and Data Preparation
ggplot(df, aes(x = G2, y = G3, color = sex)) + 
  geom_point() + 
  ggtitle("G2 vs G3 by Sex") +
  theme_minimal()

# Check feature importance using a simple linear model
lm_model <- lm(G3 ~ G1 + G2 + sex + age + address + Pstatus + Medu + Fedu + Mjob + Fjob + reason + schoolsup, data = df)

print("Linear Model Summary:")
print(summary(lm_model))

# Feature selection using Recursive Feature Elimination
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfe_result <- rfe(df %>% select(-G3), df$G3, sizes = c(1:5), rfeControl = control)

print("Recursive Feature Elimination Results:")
print(rfe_result)

set.seed(42)

# Split the data into training and testing
train_index <- createDataPartition(df$G3, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

print(paste("Training set size:", nrow(train_data)))
print(paste("Testing set size:", nrow(test_data)))

# Part 4: Random Forest analysis
set.seed(123)
rf_model <- randomForest(G3 ~ ., data = train_data, ntree = 100, importance = TRUE)

# Make predictions on test data
predictions <- predict(rf_model, newdata = test_data)

# Evaluate model performance
rmse <- sqrt(mean((predictions - test_data$G3)^2))
mae <- mean(abs(predictions - test_data$G3))

print(paste("RMSE:", rmse))
print(paste("MAE:", mae))

# Part 5: Model Performance
# Feature Importance Plot
importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)  

print("Feature Importance DataFrame:")
print(importance_df)

ggplot(importance_df, aes(x = reorder(Feature, IncNodePurity), y = IncNodePurity)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  ggtitle("Feature Importance (Random Forest - Node Purity)") +
  xlab("Features") +
  ylab("Increase in Node Purity")

# Actual vs. Predicted Plot
ggplot(data.frame(Actual = test_data$G3, Predicted = predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  ggtitle("Actual vs. Predicted Test Scores") +
  xlab("Actual G3 Scores") +
  ylab("Predicted G3 Scores")
