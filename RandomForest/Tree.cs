namespace RandomForest 
{ 
    public class Tree : IModel
    {
        private int? MaxDepth { get; }
        private Node Root { get; set; }
        private double[][] Data { get; set; }
        private int[] Targets { get; set; }

        /// <summary>
        /// Constructor to initialize the Tree.
        /// </summary>
        /// <param name="maxDepth">Maximum depth of the tree.</param>
        /// <param name="treeRoot">Root node of the tree.</param>
        /// <param name="data">Training data.</param>
        /// <param name="targets">Target labels.</param>
        public Tree(int? maxDepth,
                    Node treeRoot,
                    double[][] data,
                    int[] targets) {
            MaxDepth = maxDepth;
            Root = treeRoot;
            Data = data;
            Targets = targets;
        }

        /// <summary>
        /// Trains the tree model on the given data and targets.
        /// </summary>
        /// <param name="data">The training data as a 2D array of features.</param>
        /// <param name="targets">The target labels corresponding to the training data.</param>
        public void Fit(double[][] data, int[] targets) {
            Data = data;
            Targets = targets;
            var dataIndices = Enumerable.Range(0, data.Length).ToList();
            Root = new Node(dataIndices, 
                RandomForestHelper.MostCommonClass(dataIndices, targets));
            BuildTree();
        }

        /// <summary>
        /// Predicts the target labels for the given data.
        /// </summary>
        /// <param name="data">The data to predict on as a 2D array of features.</param>
        /// <returns>An array of predicted labels.</returns>
        public int[] Predict(double[][] data) {
            return PredictInput(data);
        }

        /// <summary>
        /// Method to build the tree recursively.
        /// </summary>
        private void BuildTree() {
            SplitWithRecursion(Root, 0);
        }

        /// <summary>
        /// Recursive method to split nodes and build the tree.
        /// </summary>
        /// <param name="node">Current node to split.</param>
        /// <param name="depth">Current depth of the tree.</param>
        private void SplitWithRecursion(Node node, int depth)
        {
            double nodeCriterion = Entropy(Targets.Where((_, idx) => node.DataIndices.Contains(idx)).ToArray());

            if (!node.CanSplit(depth, MaxDepth, nodeCriterion)) {
                return;
            }

            (int? feature, double? value, List<int>? left, List<int>? right, double? criterion) =
                BestSplit(node.DataIndices);

            if (!feature.HasValue || !value.HasValue || left == null || right == null || !criterion.HasValue) 
                return;
            
            var leftNode = new Node(left, RandomForestHelper.MostCommonClass(left, Targets));
            var rightNode = new Node(right, RandomForestHelper.MostCommonClass(right, Targets));
            node.Split(feature.Value, value.Value, leftNode, rightNode);
            // Console.WriteLine($"{node.DataIndices.Count}, {leftNode.DataIndices.Count},
            // {rightNode.DataIndices.Count}, ---ENTROPY: {Entropy(Targets.Where((_, idx)
            // => node.DataIndices.Contains(idx)).ToArray())}---");
            var newDepth = depth + 1;
            SplitWithRecursion(leftNode, newDepth);
            SplitWithRecursion(rightNode, newDepth);
        }

        /// <summary>
        /// Method to find the best split for the node.
        /// </summary>
        /// <param name="instances">List of data indices at the node.</param>
        /// <returns>Tuple containing the best feature, value, left and right instances, and criterion.</returns>
        private (int? bestFeature,
                double? bestValue,
                List<int>? bestLeftInstances,
                List<int>? bestRightInstances,
                double? bestCriterion) BestSplit(List<int> instances)
        {
            var bestCriterion = double.MaxValue;
            int? bestFeature = null;
            double? bestValue = null;
            List<int>? bestLeftInstances = null;
            List<int>? bestRightInstances = null;
            
            var features = RandomForestHelper.SubSampleFeatures(Data[0].Length);

            foreach (var feature in features)
            {
                double[] uniqueSortedValues = Data.Where((x, idx) => instances.Contains(idx))
                    .Select(d => d[feature])
                    .Distinct()
                    .OrderBy(x => x)
                    .ToArray();
                
                double[] midpoints = new double[uniqueSortedValues.Length - 1];
                for (var i = 0; i < uniqueSortedValues.Length - 1; ++i) {
                    midpoints[i] = (uniqueSortedValues[i] + uniqueSortedValues[i + 1]) / 2;
                }

                foreach (var value in midpoints)
                {
                    var leftInstances = instances.Where(idx => Data[idx][feature] <= value).ToList();
                    var rightInstances = instances.Where(idx => Data[idx][feature] > value).ToList();

                    if (leftInstances.Count <= 0 || rightInstances.Count <= 0) 
                        continue;
                
                    var currentCriterion = Entropy(Targets.Where((_, idx) =>
                                               leftInstances.Contains(idx)).ToArray()) +
                                           Entropy(Targets.Where((_, idx) =>
                                               rightInstances.Contains(idx)).ToArray());

                    if (!(currentCriterion < bestCriterion)) 
                        continue;
                    
                    bestCriterion = currentCriterion;
                    bestFeature = feature;
                    bestValue = value;
                    bestLeftInstances = leftInstances;
                    bestRightInstances = rightInstances;
                }
            }
            
            return (bestFeature, bestValue, bestLeftInstances, bestRightInstances, bestCriterion);
        }
        
        /// <summary>
        /// Method to calculate the entropy of the targets.
        /// Tells us how imprune the node is.
        /// If all the instances belong to the same class, the imprunity level will be 0
        /// if all the instances belong to all different class, the imprunity level will be 1
        /// </summary>
        /// <param name="targets">Array of target labels.</param>
        /// <returns>The entropy value.</returns>
        private double Entropy(int[] targets) {
            int maxClass = targets.Max();
            var bins = Enumerable.Range(0, maxClass + 1)
                .Select(i => targets.Count(t => t == i))
                .ToArray();
    
            int total = targets.Length;
            double entropy = 0.0;

            foreach (int count in bins)
            {
                if (count <= 0) 
                    continue;
                
                double p = (double)count / total;
                entropy -= count * Math.Log(p);
            }
            return entropy;
        }
        
        /// <summary>
        /// Method to predict the class labels for the input data.
        /// </summary>
        /// <param name="data">The input data as a 2D array.</param>
        /// <returns>An array of predicted class labels.</returns>
        private int[] PredictInput(double[][] data)
        {
            int[] predictions = new int[data.Length];

            for (int example = 0; example < data.Length; example++)
            {
                Node? node = Root;

                while (node is { IsLeaf: false })
                {
                    if (node.SplitFeatureIndex == null || node.SplitValue == null) {
                        throw new InvalidOperationException("Split feature index or split " +
                                                            "value is not set for a non-leaf node.");
                    }

                    if (data[example][node.SplitFeatureIndex.Value] <= node.SplitValue) {
                        node = node.Left;
                    }
                    else {
                        node = node.Right;
                    }
                }

                if (node != null)
                    predictions[example] = node.PredictedClass;
            }

            return predictions;
        }
        
        /// <summary>
        /// Debugs the tree by printing information about each node.
        /// </summary>
        public void DebugTree()
        {
            TraverseAndDebug(Root, 0);
        }

        /// <summary>
        /// Recursively traverses the tree and prints debug information about each node.
        /// </summary>
        /// <param name="node">The current node being traversed.</param>
        /// <param name="depth">The current depth of the node in the tree.</param>
        private void TraverseAndDebug(Node node, int depth)
        {
            if (node == null)
            {
                return;
            }

            // Print the current node's information
            var indent = new string(' ', depth * 2); // Indentation based on depth
            Console.WriteLine($"{indent}Node at depth {depth}:");
            if (node.IsLeaf)
            {
                Console.WriteLine($"{indent}  Leaf node - Predicted class: {node.PredictedClass}");
            }
            else
            {
                Console.WriteLine($"{indent}  Internal node - Split feature: {node.SplitFeatureIndex}, Split value: {node.SplitValue}");
                Console.WriteLine($"{indent}  Left child:");
                if (node.Left != null) TraverseAndDebug(node.Left, depth + 1);
                Console.WriteLine($"{indent}  Right child:");
                if (node.Right != null) TraverseAndDebug(node.Right, depth + 1);
            }
        }
    }
}
