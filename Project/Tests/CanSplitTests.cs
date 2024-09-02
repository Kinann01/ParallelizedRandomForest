using RandomForest;
using Xunit;

namespace TestProject1
{
    public class NodeTests
    {
        [Fact]
        public void TestCanSplit_WhenNodeCanBeSplit_ReturnsTrue()
        {
            // Arrange
            var dataIndices = new List<int> { 0, 1, 2, 3 };
            var node = new Node(dataIndices, 0);

            int currentDepth = 2;
            int? maxDepth = 5;
            double criterion = 1.0;

            // Act
            bool canSplit = node.CanSplit(currentDepth, maxDepth, criterion);

            // Assert
            Assert.True(canSplit);
        }

        [Fact]
        public void TestCanSplit_WhenMaxDepthReached_ReturnsFalse()
        {
            // Arrange
            var dataIndices = new List<int> { 0, 1, 2, 3 };
            var node = new Node(dataIndices, 0);

            int currentDepth = 5;
            int? maxDepth = 5;
            double criterion = 1.0;

            // Act
            bool canSplit = node.CanSplit(currentDepth, maxDepth, criterion);

            // Assert
            Assert.False(canSplit);
        }

        [Fact]
        public void TestCanSplit_WhenCriterionIsZero_ReturnsFalse()
        {
            // Arrange
            var dataIndices = new List<int> { 0, 1, 2, 3 };
            var node = new Node(dataIndices, 0);

            int currentDepth = 2;
            int? maxDepth = 5;
            double criterion = 0.0;

            // Act
            bool canSplit = node.CanSplit(currentDepth, maxDepth, criterion);

            // Assert
            Assert.False(canSplit);
        }

        [Fact]
        public void TestCanSplit_WhenNoMaxDepthAndPositiveCriterion_ReturnsTrue()
        {
            // Arrange
            var dataIndices = new List<int> { 0, 1, 2, 3 };
            var node = new Node(dataIndices, 0);

            int currentDepth = 2;
            int? maxDepth = null;
            double criterion = 1.0;

            // Act
            bool canSplit = node.CanSplit(currentDepth, maxDepth, criterion);

            // Assert
            Assert.True(canSplit);
        }
    }
}
