using RandomForest;
using Xunit;

namespace TestProject1
{
    public class NodeSplitTests
    {
        [Fact]
        public void TestSplit_CorrectlySetsChildNodesAndUpdatesState()
        {
            // Arrange
            var dataIndices = new List<int> { 0, 1, 2, 3 };
            var node = new Node(dataIndices, 0);

            int featureIndex = 1;
            double splitValue = 0.5;
            var leftNode = new Node(new List<int> { 0, 1 }, 0);
            var rightNode = new Node(new List<int> { 2, 3 }, 1);

            // Act
            node.Split(featureIndex, splitValue, leftNode, rightNode);

            // Assert
            Assert.False(node.IsLeaf);
            Assert.Equal(featureIndex, node.SplitFeatureIndex);
            Assert.Equal(splitValue, node.SplitValue);
            Assert.Same(leftNode, node.Left);
            Assert.Same(rightNode, node.Right);
            Assert.Equal(-1, node.PredictedClass);  // Should be reset to -1 after split
        }

        [Fact]
        public void TestSplit_CorrectlySplitsNodeWithEmptyChildren()
        {
            // Arrange
            var dataIndices = new List<int> { 0, 1, 2, 3 };
            var node = new Node(dataIndices, 0);

            int featureIndex = 1;
            double splitValue = 0.5;
            var leftNode = new Node(new List<int>(), -1);
            var rightNode = new Node(new List<int>(), -1);

            // Act
            node.Split(featureIndex, splitValue, leftNode, rightNode);

            // Assert
            Assert.False(node.IsLeaf);
            Assert.Equal(featureIndex, node.SplitFeatureIndex);
            Assert.Equal(splitValue, node.SplitValue);
            Assert.Same(leftNode, node.Left);
            Assert.Same(rightNode, node.Right);
        }
    }
}