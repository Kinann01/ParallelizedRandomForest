namespace RandomForest
{
    /// <summary>
    /// Interface for AI models with basic methods for training and prediction.
    /// </summary>
    public interface IModel
    {
        /// <summary>
        /// Trains the model on the given data and targets.
        /// </summary>
        /// <param name="data">The training data as a 2D array of features.</param>
        /// <param name="targets">The target labels corresponding to the training data.</param>
        void Fit(double[][] data, int[] targets);

        /// <summary>
        /// Predicts the target labels for the given data.
        /// </summary>
        /// <param name="data">The data to predict on as a 2D array of features.</param>
        /// <returns>An array of predicted labels.</returns>
        int[] Predict(double[][] data);
    }
}