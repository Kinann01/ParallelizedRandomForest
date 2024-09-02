# Parallelized Random Forest

### Overview

- This project implements a Parallelized Random Forest classifier in C#. The classifier uses decision trees as its base learners and aggregates their predictions to make final predictions. The goal is to efficiently train and predict with multiple decision trees. We take advantage of the parallel computing to improve efficiency. The parallelization is achieved through a custom thread pool implementation that allows for efficient task scheduling and execution. The project acts as a library for users to import and use. The main two usage methods are `Fit` and `Predict`. An example will be provieded later. 

- Moreover the implementation is done in a way where simply decision trees can also be imported and used. This is done as both the `Decision Tree` class and the `Random Forest` class extend an `IModel` interface. This interface has two methods, `Fit` and `Predict`.

### Key Features

- Random Forest Classifier: An ensemble method that combines multiple decision tres to improve predictive accuracy and control overfitting of the model.

- Decision Trees: Since decision trees are the building blocks of the random forest, they are also implemented as part of the project. The trees are grown to a maximum depth if specified or until the data at a node is pure.

- BestSplit Method: A critical function that determines the optimal feature and threshold for splitting the data at each node in the tree.

- Custom Thread Pool: Implements the thread pool design pattern to manage and execute tasks concurrently. 

- Parallelized Training and Prediction: The construction of decision trees and the prediction of results are parallelized. These tasks are done synchronously since they are independent of each other. 


### Main Functionality 

The most important part is the `BestSplit` method. The `BestSplit` method ensures that a node is split correctly. It finds the best feature and threshold to split the data at a node. The method calculates the Entropy of the data before and after the split and selects the feature and threshold/value that minimizes the impurity. The method is used to grow the decision trees in the random forest. Here is an overview:

1. **Feature Selection**: The method selects features to consider for splitting. This selection can be randomized to improve generalization and reduce correlations between trees. The randomization is done through the 'FeatureSubSampling' method. 

2. **Midpoint Calculation**: For each selected feature, unique sorted values are considered, and midpoints between these values are calculated as potential split points. We take unique midpoints to minimize the number of potential splits.

3. **Criterion Calculation**: For each potential split, the method calculates a criterion, in our case the `entropy`, to measure how well the split separates the data. The goal is to minimize this criterion.

4. **Best Split Determination**: The feature and value combination that provides the lowest criterion is chosen as the best split, and the data is divided into left and right subsets accordingly.

### Custom Thread Pool 

A custom thread pool is implemented to handle parallel execution of tasks efficiently. The thread pool is used to manage a pool of threads that execute tasks concurrently. Here is how it works: 

1. **Task Queue** : Tasks (in this case, training a tree or predicting with a tree) are enqueued into a task queue.

2. **Worker Threads**: A fixed number of worker threads are created at the start. These threads continuously pull tasks from the queue and execute them. The number of threads is configured to be equivalent to the number of cores on the machine using `Environment.ProcessorCount`.

3. **Task Execution**: Each thread works on a task from the queue. Once a task is completed, the thread moves on to the next task in the queue.

4. **Shutdown Mechanism**: Once all tasks are completed, the thread pool is shut down, ensuring that all threads have completed their work before the program exits.

5. **Thread Safety**: The thread pool is designed to be thread-safe, ensuring that tasks are executed correctly and efficiently.


### Parallelized Training and Prediction 


Training

- Fit Method: The Fit method trains the Random Forest by growing multiple decision trees in parallel. Each tree is built on a different subset of the data (using bootstrap sampling if bagging is enabled) and a subset of features.

- Parallelization: A thread pool is used to train each tree in parallel.

Prediction

- Predict Method: The Predict method aggregates predictions from all the decision trees in the forest. The final prediction is made based on the majority vote across all trees.

- Parallel Prediction: Predictions from individual trees are computed in parallel using the thread pool.

### Installation and how to use it 

- You will have to clone the repository first 

```bash
git clone https://github.com/Kinann01/ParallelizedRandomForest
```

- Then you will have to import the project into your solution and add a reference to it. 

- You can then use the RandomForest class in your code. Here is an example of how to use the RandomForest class:

```C#
using RandomForest;
using CustomThreadPool;

class Program
{
    static void Main(string[] args)
    {
        var forest = new RandomForest(10, true, 5);
        // Load your data
        double[][] trainData = ...;
        int[] trainTargets = ...;

        // Train the RandomForest
        forest.Fit(trainData, trainTargets);

        // Make predictions
        double[][] testData = ...;
        var predictions = forest.Predict(testData);

        // Output or evaluate predictions
    }
}
```