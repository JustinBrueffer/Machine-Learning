using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord;
using Accord.MachineLearning;
using Accord.Controls;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics;
using Accord.Statistics.Kernels;

namespace XOR_Gatter
{
    class Program
    {
        [MTAThread]
        static void Main(string[] args)
        {
            double[][] inputs =
            {
                /* 1. */ new double[] { 0, 0 },
                /* 2. */ new double[] { 1, 0 },
                /* 3. */ new double[] { 0, 1 },
                /* 4. */ new double[] { 1, 1 }
            };

            int[] outputs =
            {
                /* 1. 0 xor 0 = 0: */ 1,
                /* 2. 1 xor 0 = 1: */ 0,
                /* 3. 0 xor 1 = 1: */ 0,
                /* 4. 1 xor 1 = 0: */ 0,
            };

            var smo = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100
            };

            var svm = smo.Learn(inputs, outputs);

            bool[] prediction = svm.Decide(inputs);

            double error = new AccuracyLoss(outputs).Loss(prediction);

            Console.WriteLine("Error: " + error);

            ScatterplotBox.Show("Training data", inputs, outputs);
            ScatterplotBox.Show("SVM results", inputs, prediction.ToZeroOne());

            Console.ReadKey();
        }
    }
}
