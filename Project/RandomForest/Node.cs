namespace RandomForest
{
    public class Node
    {
        public bool IsLeaf { get; private set; }
        public List<int> DataIndices { get; private set; } // Indices of data points
        public int? SplitFeatureIndex { get; private set; } // Index of the feature used for splitting
        public double? SplitValue { get; private set; }     // Value of the feature for splitting
        public int PredictedClass { get; private set; }    // Prediction for leaf nodes
        public Node? Left { get; private set; }             // Left child node
        public Node? Right { get; private set; }            // Right child node

        
        /// <summary>
        /// Initializes a new instance of the <see cref="Node"/> class.
        /// </summary>
        /// <param name="dataIndices">The indices of the data points associated with the node.</param>
        /// <param name="predictedClass">The predicted class for the node.</param>
        public Node(List<int> dataIndices, int predictedClass)
        {
            IsLeaf = true;
            DataIndices = dataIndices ?? new List<int>();
            PredictedClass = predictedClass;
            SplitFeatureIndex = null;
            SplitValue = null;
            Left = null;
            Right = null;
        }

        /// <summary>
        /// Splits the node into two child nodes.
        /// </summary>
        /// <param name="splitFeatureIndex">The index of the feature used for splitting.</param>
        /// <param name="splitValue">The value of the feature for splitting.</param>
        /// <param name="left">The left child node.</param>
        /// <param name="right">The right child node.</param>
        public void Split(int splitFeatureIndex, double splitValue, Node left, Node right)
        {
            IsLeaf = false;
            SplitFeatureIndex = splitFeatureIndex;
            SplitValue = splitValue;
            Left = left;
            Right = right;
            PredictedClass = -1;
        }

        /// <summary>
        /// Determines whether the node can be split based on the given criteria.
        /// </summary>
        /// <param name="depth">The current depth of the node.</param>
        /// <param name="maxDepth">The maximum allowable depth.</param>
        /// <param name="criterion">The splitting criterion value.</param>
        /// <returns><c>true</c> if the node can be split; otherwise, <c>false</c>.</returns>
        public bool CanSplit(int depth, int? maxDepth, double criterion)
        {
            return (!maxDepth.HasValue || depth < maxDepth.Value) && criterion != 0;
        }
    }
}