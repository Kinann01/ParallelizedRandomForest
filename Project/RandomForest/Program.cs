namespace RandomForest
{
    using System;
    using System.Linq;
    using CsvHelper;
    using CsvHelper.Configuration;
    using System.Globalization;
    using System.IO;
    using C = Console;

    /// <summary>
    /// Utility class for reading CSV files.
    /// </summary>
    public abstract class MyCsvReader
    {
        /// <summary>
        /// Reads data from a CSV file into a 2D array of doubles.
        /// </summary>
        /// <param name="filePath">The path to the CSV file.</param>
        /// <returns>A 2D array of doubles representing the data.</returns>
        public static double[][] ReadData(string filePath)
        {
            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
                   {
                       HasHeaderRecord = true,
                       MissingFieldFound = null
                   }))
            {
                var records = csv.GetRecords<dynamic>().ToList();
                var header = csv.Context.Reader!.HeaderRecord;
                int featureCount = header!.Length;

                double[][] data = new double[records.Count][];

                for (int i = 0; i < records.Count; i++)
                {
                    data[i] = new double[featureCount];
                    var record = (IDictionary<string, object>)records[i];
                    for (int j = 0; j < featureCount; j++)
                    {
                        data[i][j] = Convert.ToDouble(record[header[j]], CultureInfo.InvariantCulture);
                    }
                }

                return data;
            }
        }

        /// <summary>
        /// Reads target labels from a CSV file into an array of integers.
        /// </summary>
        /// <param name="filePath">The path to the CSV file.</param>
        /// <returns>An array of integers representing the target labels.</returns>
        public static int[] ReadTargets(string filePath)
        {
            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
                   {
                       HasHeaderRecord = true,
                       MissingFieldFound = null
                   }))
            {
                var records = csv.GetRecords<int>().ToList();
                return records.ToArray();
            }
        }
    }
    
    /// <summary>
    /// Main program class for running the RandomForest example.
    /// </summary>
    public abstract class Program
    {
        /// <summary>
        /// The entry point of the program.
        /// </summary>
        /// <param name="args">The command-line arguments.</param>
        public static void Main(string[] args)
        {
            // Training Data
            string trainDataFilePath = "/Users/kinanal-falakh/Desktop/Project/RandomForest/data/wine_train_data.csv";
            string trainTargetsFilePath = "/Users/kinanal-falakh/Desktop/Project/RandomForest/data/wine_train_target.csv";
            
            var trainingData = MyCsvReader.ReadData(trainDataFilePath);
            var trainingTargets = MyCsvReader.ReadTargets(trainTargetsFilePath);

            // Testing Data
            string testDataFilePath = "/Users/kinanal-falakh/Desktop/Project/RandomForest/data/wine_test_data.csv";
            string testTargetFilePath = "/Users/kinanal-falakh/Desktop/Project/RandomForest/data/wine_test_target.csv";

            var testingData = MyCsvReader.ReadData(testDataFilePath);
            var testingTargets = MyCsvReader.ReadTargets(testTargetFilePath);

            // Initialize the RandomForest model using the IModel interface
            IModel forest = new RandomForest(3, false, 3);

            // Train the RandomForest
            forest.Fit(trainingData, trainingTargets);

            // Predict on training and testing data
            var trainPrediction = forest.Predict(trainingData);
            var testPrediction = forest.Predict(testingData);

            // Calculate accuracy
            double trainAccuracy = RandomForestHelper.CalculateAccuracy(trainingTargets,
                trainPrediction);
            double testAccuracy = RandomForestHelper.CalculateAccuracy(testingTargets, 
                testPrediction);

            // Output results
            C.WriteLine($"Training Accuracy: {trainAccuracy * 100:F2}%");
            C.WriteLine($"Testing Accuracy: {testAccuracy * 100:F2}%");
        }
    }
}
