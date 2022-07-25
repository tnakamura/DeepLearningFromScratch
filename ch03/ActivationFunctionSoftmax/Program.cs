using NumSharp;

Console.WriteLine(nameof(StepFunction));
{
    var x = np.array(-1.0, 1.0, 2.0);
    var y = StepFunction(x);
    Console.WriteLine(y.ToString());
}

Console.WriteLine(nameof(Sigmoid));
{
    var x = np.array(-1.0, 1.0, 2.0);
    var y = Sigmoid(x);
    Console.WriteLine(y.ToString());
}

Console.WriteLine(nameof(ReLU));
{
    var x = np.array(-1.0, 1.0, 2.0);
    var y = ReLU(x);
    Console.WriteLine(y.ToString());
}

Console.WriteLine(nameof(Softmax));
{
    var x = np.array(0.3, 2.9, 4.0);
    var y = Softmax(x);
    Console.WriteLine(y.ToString());
    Console.WriteLine(np.sum(y, NPTypeCode.Double).ToString());
}

Console.ReadLine();

static NDArray StepFunction(NDArray x)
{
    var y = x > 0.0;
    return y.astype(NPTypeCode.Int32);
}

static NDArray Sigmoid(NDArray x)
{
    return 1 / (1 + np.exp(x * -1));
}

static NDArray ReLU(NDArray x)
{
    return np.maximum(0.0, x);
}

static NDArray Softmax(NDArray a)
{
    var c = np.max(a);
    var exp_a = np.exp(a - c);
    var sum_exp_a = np.sum(exp_a, NPTypeCode.Double);
    var y = exp_a / sum_exp_a;
    return y;
}

