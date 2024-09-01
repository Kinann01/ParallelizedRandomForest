namespace RandomForest
{
    using System;
    using System.Linq;

    /// <summary>
    /// Helper methods for the RandomForest implementation.
    /// </summary>
    public static class RandomForestHelper
    {
        private static readonly MersenneTwister GeneratorFeatureSubsampling = new(44);
        private static readonly MersenneTwister GeneratorBootstrapping = new(44);

        /// <summary>
        /// Sub-samples features from the given number of features.
        /// </summary>
        /// <param name="numberOfFeatures">The total number of features.</param>
        /// <param name="featureSubsampling">The fraction of features to subsample.</param>
        /// <returns>An array of sub-sampled feature indices.</returns>
        public static int[] SubSampleFeatures(int numberOfFeatures, double featureSubsampling = 1.0)
        {
            int sampleSize = (int)(featureSubsampling * numberOfFeatures);
            var features = Enumerable.Range(0, numberOfFeatures).ToArray();

            // Sample without replacement
            return features.OrderBy(_ => GeneratorFeatureSubsampling.NextUInt()).Take(sampleSize).OrderBy(x => x).ToArray();
        }

        /// <summary>
        /// Creates a bootstrap sample of the dataset.
        /// </summary>
        /// <param name="trainData">The training data as a 2D array.</param>
        /// <returns>An array of indices representing the bootstrap sample.</returns>
        public static int[] BootstrapDataset(double[][] trainData)
        {
            int dataSize = trainData.Length;
            var indices = new int[dataSize];
            for (int i = 0; i < dataSize; i++)
            {
                // Sample with replacement
                indices[i] = GeneratorBootstrapping.Next(dataSize);
            }
            return indices;
        }

        /// <summary>
        /// Calculates the accuracy of predictions.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The accuracy as a double.</returns>
        public static double CalculateAccuracy(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
            {
                throw new ArgumentException("The length of true labels and predicted labels must be the same.");
            }

            int correctCount = trueLabels.Zip(predictedLabels, (trueLabel, predictedLabel)
                => trueLabel == predictedLabel).Count(match => match);
            return (double)correctCount / trueLabels.Length;
        }

        /// <summary>
        /// Method to find the most common class in the instances.
        /// </summary>
        /// <param name="instances">List of data indices.</param>
        /// <param name="targets">Targets of the dataset to compare the count of classes.</param>
        /// <returns>The most common class label.</returns>
        public static int MostCommonClass(List<int> instances, int[] targets)
        {
            return targets
                .Where((_, idx) => instances.Contains(idx)) // Filter targets by indices in instances
                .GroupBy(t => t) // Group by target value
                .OrderByDescending(g => g.Count()) // Order groups by count
                .First() // Take the first group (most common class)
                .Key; // Return the class label (key)
        }
    }

    /// <summary>
    /// Implementation of the Mersenne Twister random number generator.
    /// </summary>
    public class MersenneTwister
    {
        private const int N = 624;
        private const int M = 397;
        private const uint MatrixA = 0x9908b0df;
        private const uint UpperMask = 0x80000000;
        private const uint LowerMask = 0x7fffffff;
        private readonly uint[] mt = new uint[N];
        private int mti;

        /// <summary>
        /// Initializes a new instance of the <see cref="MersenneTwister"/> class with a given seed.
        /// </summary>
        /// <param name="seed">The seed for the random number generator.</param>
        public MersenneTwister(uint seed)
        {
            mt[0] = seed;
            for (mti = 1; mti < N; mti++)
            {
                mt[mti] = (uint)(1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
            }
        }

        /// <summary>
        /// Generates a random unsigned integer.
        /// </summary>
        /// <returns>A random unsigned integer.</returns>
        public uint NextUInt()
        {
            uint y;
            uint[] mag01 = { 0x0, MatrixA };

            if (mti >= N)
            {
                int kk;

                if (mti == N + 1)
                {
                    mt[0] = 5489;
                }

                for (kk = 0; kk < N - M; kk++)
                {
                    y = (mt[kk] & UpperMask) | (mt[kk + 1] & LowerMask);
                    mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
                }

                for (; kk < N - 1; kk++)
                {
                    y = (mt[kk] & UpperMask) | (mt[kk + 1] & LowerMask);
                    mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
                }

                y = (mt[N - 1] & UpperMask) | (mt[0] & LowerMask);
                mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

                mti = 0;
            }

            y = mt[mti++];

            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= (y >> 18);

            return y;
        }

        /// <summary>
        /// Generates a random integer in the range [0, maxValue).
        /// </summary>
        /// <param name="maxValue">The exclusive upper bound of the random number.</param>
        /// <returns>A random integer in the range [0, maxValue).</returns>
        public int Next(int maxValue)
        {
            return (int)(NextUInt() % (uint)maxValue);
        }
    }
}
