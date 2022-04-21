# Load The Data
library(datasets)
data(iris)
head(iris)
summary(iris)
df = iris[-5]
head(df)

# Elbow method
pkgs <- c("factoextra",  "NbClust")
#install.packages(pkgs)
library(factoextra)
library(NbClust)
fviz_nbclust(df, kmeans, method = "wss") + 
  geom_vline(xintercept = 3, linetype = 2) + 
    labs(subtitle = "Elbow method")

print(" Since the Elbow occurs at 3... Hence 3 is the optimum number of clusters")

# Installing Packages
# install.packages("ClusterR")
# install.packages("cluster")

# Loading package
library(ClusterR)
library(cluster)

# Fitting K-Means clustering Model to training dataset
set.seed(240) # Setting seed
kmeans.re <- kmeans(df, centers = 3, nstart = 20)
kmeans.re

# Cluster identification for each observation
kmeans.re$cluster

# Confusion Matrix
cm <- table(iris$Species, kmeans.re$cluster)
cm

# Model Evaluation and visualization
plot(df[c("Sepal.Length", "Sepal.Width")])
plot(df[c("Sepal.Length", "Sepal.Width")],
     col = kmeans.re$cluster)
plot(df[c("Sepal.Length", "Sepal.Width")],
     col = kmeans.re$cluster,
     main = "K-means with 3 clusters")

## Plotiing cluster centers
kmeans.re$centers
kmeans.re$centers[, c("Sepal.Length", "Sepal.Width")]
points(kmeans.re$centers[, c("Sepal.Length", "Sepal.Width")],
       col = 1:3, pch = 8, cex = 3)

## Visualizing clusters
y_kmeans <- kmeans.re$cluster
clusplot(df[, c("Sepal.Length", "Sepal.Width")],
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = TRUE,
         span = TRUE,
         main = paste("Cluster iris"),
         xlab = 'Sepal.Length',
         ylab = 'Sepal.Width')

