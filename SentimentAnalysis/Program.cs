using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _zippedModel = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            // Create ML.NET context/local environment - allows you to add steps in order to keep everything together 
            // during the learning process.  
            //Create ML Context with seed for repeatable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // The TextLoader loads a dataset with comments and corresponding postive or negative sentiment. 
            // When you create a loader, you specify the schema by passing a class to the loader containing
            // all the column names and their types. This is used to create the model, and train it. 
            // Initialize our TextLoader
            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            });

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            Predict(mlContext, model);

            PredictWithModelLoadedFromFile(mlContext);

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            //Note that this case, loading your training data from a file, 
            //is the easiest way to get started, but ML.NET also allows you 
            //to load data from databases or in-memory collections.
            IDataView dataView = _textLoader.Read(dataPath);

            // Create a flexible pipeline (composed by a chain of estimators) for creating/training the model.
            // This is used to format and clean the data.  
            // Convert the text column to numeric vectors (Features column) 
            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                    // Adds a FastTreeBinaryClassificationTrainer, the decision tree learner for this project  
                    .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            // Create and train the model based on the dataset that has been loaded, transformed.
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            // Returns the model we trained to use for evaluation.
            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // Evaluate the model and show accuracy stats
            // Load evaluation/test data
            IDataView dataView = _textLoader.Read(_testDataPath);

            //Take the data in, make transformations, output the data. 
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            var predictions = model.Transform(dataView);

            // BinaryClassificationContext.Evaluate returns a BinaryClassificationEvaluator.CalibratedResult
            // that contains the computed overall metrics.
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            // The Accuracy metric gets the accuracy of a classifier, which is the proportion 
            // of correct predictions in the test set.

            // The Auc metric gets the area under the ROC curve.
            // The area under the ROC curve is equal to the probability that the classifier ranks
            // a randomly chosen positive instance higher than a randomly chosen negative one
            // (assuming 'positive' ranks higher than 'negative').

            // The F1Score metric gets the classifier's F1 score.
            // The F1 score is the harmonic mean of precision and recall:
            //  2 * precision * recall / (precision + recall).

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            // Save the new model to .ZIP file
            SaveModelAsFile(mlContext, model);

        }

        private static void Predict(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = model.MakePredictionFunction<SentimentData, SentimentPrediction>(mlContext);
            
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This is a very rude movie"
            };

            var resultprediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {resultprediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void PredictWithModelLoadedFromFile(MLContext mlContext)
        {
            // Adds some comments to test the trained model's predictions.
            IEnumerable<SentimentData> sentiments = new[]
            {
                
                new SentimentData
                {
                    SentimentText = "I'm so happy"
                },
                new SentimentData
                {
                    SentimentText = "This is a very rude movie"
                },
                new SentimentData
                {
                    SentimentText = "He is a good person but have troubles sometimes."
                },
                new SentimentData
                {
                    SentimentText = "This is an awesome movie and you should see it."
                },
                new SentimentData
                {
                    SentimentText = "This is an awesome"
                },
                new SentimentData
                {
                    SentimentText = "This is an awesome movie"
                }
            };

            ITransformer loadedModel;
            using (var stream = new FileStream(_zippedModel, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // Create prediction engine
            var sentimentStreamingDataView = mlContext.CreateStreamingDataView(sentiments);
            var predictions = loadedModel.Transform(sentimentStreamingDataView);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            var predictedResults = predictions.AsEnumerable<SentimentPrediction>(mlContext, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");
            Console.WriteLine();

            // Builds pairs of (sentiment, prediction)
            var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {item.prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
            Console.ReadKey();
        }

        // Saves the model we trained to a zip file.
        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_zippedModel, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _zippedModel);
        }

    }
}
