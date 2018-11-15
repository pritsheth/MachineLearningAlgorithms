Welcome to the K means clustering wiki!



![Python](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png "Written in Python")<br >




# K Means clustering for breast cancer data:

In K means clustering, we will compute the distance of each data point with the centroid and can choose the nearest centroid to the data point. Based on different K value we can decide for which K value what is the best value of potential function. 

# Task of building the clustering algo for breast cancer data:

The data set contains 11 columns, separated by comma. The first column is the example id, and we can ignore it. The second to tenth columns are the 9 features based on which we run K means algorithm
You can manually download the dataset from UCI ML webpage (http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
or automatically get it from some libraries/packages  (e.g., as done in section 5.9 of http://scikitlearn.org/stable/datasets/index.html for sklearn in Python).

				Test Id, 10 features, Class label

Below is the size of given Input dataset : 

Train: 699 data points 

Algorithm :

- (Pre-processing part) Load csv file into numpy array and delete the invalid column. 

- For each K values run following K means Algorithm:
    
      - Start :
            - Randomly initialize K centroid.     
            - Forming cluster: 
                                - For each data point compute the distance with all the centroids
                                - Add the data point to clostest cluster
            - Repeat until convergence step:
                                - For each cluster take mean of all data points and recompute the centroids
                                - Again reform the cluster as we did previously 
                                - Stop when we have the 0 difference between old centroids and new centroids.
                                
            - Reform the centroid by taking the mean of all data points
            - Compute the potenitial function for each cluster in final list of clusters:
                                        
                                 - [subtract the data points with centroids and square it, do the 
            row-wise sum to get the actual distance and add it to final cost_value]
            
            - plot graph of K values with respect to potenital function
                             
           
            
      - End
 


Given below, results I obtained with different K values for K means :

k = [2, 3, 4, 5, 6, 7, 8]
ErrorRate = [19782.72980424593, 18538.173515170227, 15574.126207410285, 14032.417674323133, 13240.170786547589, 12973.939073163332, 12609.417122267289]

	            K = 1, with Euclidean : 
	            Error rate : 9.6 %

	            K = 9, with Euclidean :
	            Error rate : 9.1%

	            K = 19,  with Euclidean :
	            Error Rate : 10.8 % 

	            K = 29 , with Euclidean :
	            Error Rate : 12.3 %

	            K = 39, with Euclidean :
	            Error Rate : 13.4%

	            K= 49, with Euclidean :
	            Error Rate : 13.6 %

	            K = 59, with Euclidean :
	            Error Rate :  14.3 %

	            K = 69, with Euclidean :
	            Error Rate : 14.9 %
	            
	            K = 79, with Euclidean :
	            Error Rate : 15.8 %
	            
	            K = 89, with Euclidean :
	            Error Rate : 16.6 %
	            
	            K = 99, with Euclidean :
	            Error Rate : 17.1 %



# Problems faced when optimizing KNN code:

The real challenge was to compute the distance matrix for each test instance and the value of the K. It will definitely take more time if we 


# Optimizations, I did to solve above problem and better performance: 

Given the size of the input training set (about 37000) and the feature vector (192), the training for each test set will take a large amount of time. Hence, we reduced the feature set by converting them into pixel values. These are single integers formed by shifting 8 bits from initial red value adding green value and again shifting 8 bits and adding the blue value. By this method there is almost no loss in information and the feature vector is effectively reduced to 63. This did reduce time a lot than the full 192 vector, the shifting and adding values did incur an overhead and strangely it gave us very bad results on accuracy of the classifier, maybe due to the drastic differences. So we changed this and converted the values to gray pixels reducing 3:1 (red,green,blue = 1 gray pixel)

# Conclusion :
In terms of values of K When we tried picking very small values for K that is 1 or 2 then the knn classifier was over fitting the dataset. Because it was trying to build individual data points with their own individual decision boundaries. So, the classifier was performing better on training dataset that is was giving better accuracies on it whereas on test dataset the accuracy was getting reduced.

When we tried picking very large value for K that is 50 or more then the knn classifier was under fitting the dataset. Because, it is not fitting the input data properly. So, the classifier was not performing better on train as well as test dataset.


Q. If you were to pick the optimal value of K based on this curve, would you pick the one with the lowest value of the potential function? Why?

-	As we can see from graph, if we increase the value of K, we will get lesser potential function value. Ideal value of K should not be too big and too small. 
-	E.g if we take the value of K = no of data samples then we will get the 0 as the value of potential function.
-	So, optimal value of K will be that when we see the less decrease rate of the potential function. When the function value decrease sharply and after which it decrease at very slower rate. (Elbow rule)
-	In this case K = 4 (Optimized value for given data set)


# Parameters and Results :

Image encoding: Grayscale
Distance measure: Euclidean
Accuracy:  between 80 % and 91%
Execution time: 51.46 seconds

You can run the program as follows:

- Clone the repository
- Open Kmeans Classifier module
- Run KNN classifier ($ python KNNClassifier.py )
- Make sure you have the mnist_test and mnist_train csv files in the same folder. [Otherwise, run the convert method to create the csv files]
- Check the time, plotted graph, error_rate metrics
