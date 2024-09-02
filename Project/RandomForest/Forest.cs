using System;
using System.Collections.Generic;
using System.Linq;
using CustomThreadPool;

namespace RandomForest
{
    /// <summary>
    /// Represents a random forest classifier.
    /// </summary>
    public class RandomForest : IModel
    {
        /// <summary>
        /// Gets the number of trees in the forest.
        /// </summary>
        private int NumberOfTrees { get; init; }

        /// <summary>
        /// Gets the maximum depth of the trees.
        /// </summary>
        private int? MaxDepth { get; init; }

        /// <summary>
        /// Gets the list of all trees in the forest.
        /// </summary>
        private List<Tree> AllTrees { get; init; }

        /// <summary>
        /// Indicates whether bagging is used.
        /// </summary>
        private readonly bool _bagging;

        /// <summary>
        /// Initializes a new instance of the <see cref="RandomForest"/> class.
        /// </summary>
        /// <param name="numberOfTrees">The number of trees in the forest.</param>
        /// <param name="bagging">Whether to use bagging.</param>
        /// <param name="maxDepth">The maximum depth of the trees.</param>
        public RandomForest(int numberOfTrees, 
                            bool bagging, 
                            int? maxDepth)
        {
            NumberOfTrees = numberOfTrees;
            _bagging = bagging;
            MaxDepth = maxDepth;
            AllTrees = new List<Tree>(numberOfTrees);
        }

        /// <summary>
        /// Creates a decision tree.
        /// </summary>
        /// <param name="trainData">The training data.</param>
        /// <param name="trainTarget">The training target labels.</param>
        /// <returns>A trained decision tree.</returns>
        private Tree CreateTree(double[][] trainData, int[] trainTarget)
        {
            // Create the root node with the initial instances and most common class as the prediction
            var dataIndices = Enumerable.Range(0, trainData.Length).ToList();
            var root = new Node(dataIndices, RandomForestHelper.MostCommonClass(dataIndices, trainTarget)); 
            var tree = new Tree(MaxDepth, root, trainData, trainTarget);
            tree.Fit(trainData, trainTarget);
            return tree;
        }

        /// <summary>
        /// Trains the random forest.
        /// </summary>
        /// <param name="trainData">The training data.</param>
        /// <param name="trainTarget">The training target labels.</param>
        public void Fit(double[][] trainData, int[] trainTarget){
            TrainForest(trainData, trainTarget);
        }

        /// <summary>
        /// Predicts the labels for the given data using the forest of trees.
        /// </summary>
        /// <param name="data">The input data.</param>
        /// <returns>The predicted labels.</returns>
        public int[] Predict(double[][] data){
            return PredictForest(data);
        }

        /// <summary>
        /// Trains the random forest.
        /// </summary>
        /// <param name="trainData">The training data.</param>
        /// <param name="trainTarget">The training target labels.</param>
        private void TrainForest(double[][] trainData, int[] trainTarget) {
            var threadPool = new CustomThreadPool.ThreadPool(Environment.ProcessorCount);

            for (int i = 0; i < NumberOfTrees; ++i)
            {
                threadPool.EnqueueTask(() =>
                {
                    Tree tree;
                    if (_bagging) {
                        var indices = RandomForestHelper.BootstrapDataset(trainData);
                        var trainDataSubset = indices.Select(idx => trainData[idx]).ToArray();
                        var trainTargetSubset = indices.Select(idx => trainTarget[idx]).ToArray();
                        tree = CreateTree(trainDataSubset, trainTargetSubset);
                    }
                    else {
                        tree = CreateTree(trainData, trainTarget);
                    }
                    lock (AllTrees) {
                        AllTrees.Add(tree);
                    }
                });
            }
            threadPool.Shutdown();
        }

        //// <summary>
        /// Predicts the labels for the given data using the forest of trees.
        /// <param name="data">The input data.</param>
        /// <returns>The predicted labels.</returns>
        private int[] PredictForest(double[][] data)
        {
            var treePredictions = new Dictionary<int, int[]>(); // Dictionary to hold predictions from each tree
            var threadPool = new CustomThreadPool.ThreadPool(Environment.ProcessorCount);
            var syncLock = new object(); // Create a separate lock object

            // Enqueue tasks to the thread pool
            for (int i = 0; i < AllTrees.Count; ++i)
            {
                var treeIdx = i;
                threadPool.EnqueueTask(() =>
                {
                    try
                    {
                        var predictions = AllTrees[treeIdx].Predict(data);
                        lock (syncLock)
                        {
                            treePredictions[treeIdx] = predictions;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error in tree {treeIdx}: {ex.Message}");
                    }
                });
            }
            
            threadPool.Shutdown();
            var predictions = new int[data.Length];

            // Aggregate predictions from all trees
            for (int i = 0; i < data.Length; i++)
            {
                var columnPredictions = new List<int>();
                lock (syncLock) // Ensure thread-safe access to the dictionary
                {
                    foreach (var treeIdx in treePredictions.Keys) {
                        columnPredictions.Add(treePredictions[treeIdx][i]);
                    }
                }

                if (columnPredictions.Count != 0) // Check if there are any predictions
                {
                    predictions[i] = columnPredictions.GroupBy(x => x)
                                                      .OrderByDescending(g => 
                                                          g.Count()).
                                                      First().
                                                      Key;
                }
                else {
                    Console.WriteLine($"No predictions available for data point {i}. Defaulting to class 0.");
                    predictions[i] = 0; // Default to class 0 or handle this case as needed
                }
            }

            return predictions;
        }
        
        // //
        // // The following two methods are used to verify correctness of parallelization and for benchmark testing
        private int[] PredictForestWithoutParallelization(double[][] data) {
            var treePredictions = new int[AllTrees.Count][];

            // Get predictions from each tree
            for (int i = 0; i < AllTrees.Count; i++)
            {
                treePredictions[i] = AllTrees[i].Predict(data);
            }

            var predictions = new int[data.Length];

            // Aggregate predictions from all trees
            for (int i = 0; i < data.Length; i++)
            {
                var voteCount = new Dictionary<int, int>();

                for (int j = 0; j < AllTrees.Count; j++)
                {
                    int prediction = treePredictions[j][i];
                    if (!voteCount.TryAdd(prediction, 1))
                    {
                        voteCount[prediction]++;
                    }
                }

                // Get the prediction with the highest vote count
                predictions[i] = voteCount.MaxBy(kv => kv.Value).Key;
            }
            return predictions;
        }
        
        private void TrainForestWithoutParallelization(double[][] trainData, int[] trainTarget)
        {
            // Iterate over the number of trees to be created
            for (int i = 0; i < NumberOfTrees; ++i)
            {
                Tree tree;
                
                // If bagging is enabled, create a bootstrap sample of the training data
                if (_bagging) {
                    var indices = RandomForestHelper.BootstrapDataset(trainData);
                    var trainDataSubset = indices.Select(idx => trainData[idx]).ToArray();
                    var trainTargetSubset = indices.Select(idx => trainTarget[idx]).ToArray();
                    tree = CreateTree(trainDataSubset, trainTargetSubset);
                }
                else {
                    // Otherwise, use the entire training dataset
                    tree = CreateTree(trainData, trainTarget);
                }
        
                // Add the created tree to the list of trees in the forest
                AllTrees.Add(tree);
            }
        }
    }
}