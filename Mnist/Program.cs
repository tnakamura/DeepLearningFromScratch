using MnistSharp;

var (xTrain, tTrain, xTest, tTest) = await Mnist.LoadAsync(flatten: true, normalize: false);
Console.WriteLine(xTrain.Shape);
Console.WriteLine(tTrain.Shape);
Console.WriteLine(xTest.Shape);
Console.WriteLine(tTest.Shape);
Console.ReadLine();

