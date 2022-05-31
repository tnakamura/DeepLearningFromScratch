using NumSharp;

var network = InitNetwork();
var x = np.array(new double[][]
{
    new [] { 1.0, 0.5 },
});
var y = Forward(network, x);
Console.WriteLine(y.ToString());

Console.ReadLine();


static NDArray IdentityFunction(NDArray x) => x;

static NDArray Sigmoid(NDArray x) =>
    1 / (1 + np.exp(x * -1));

static IReadOnlyDictionary<string, NDArray> InitNetwork()
{
    var network = new Dictionary<string, NDArray>();

    network["W1"] = np.array(new double[][]
    {
        new [] { 0.1, 0.3, 0.5 },
        new [] { 0.2, 0.4, 0.6 },
    });
    network["b1"] = np.array(0.1, 0.2, 0.3);
    network["W2"] = np.array(new double[][]
    {
        new [] { 0.1, 0.4 },
        new [] { 0.2, 0.5 },
        new [] { 0.3, 0.6 },
    });
    network["b2"] = np.array(0.1, 0.2);
    network["W3"] = np.array(new double[][]
    {
        new [] { 0.1, 0.3 },
        new [] { 0.2, 0.4 },
    });
    network["b3"] = np.array(0.1, 0.2);

    return network;
}

static NDArray Forward(IReadOnlyDictionary<string, NDArray> network, NDArray x)
{
    var W1 = network["W1"];
    var W2 = network["W2"];
    var W3 = network["W3"];
    var b1 = network["b1"];
    var b2 = network["b2"];
    var b3 = network["b3"];

    var a1 = np.dot(x, W1) + b1;
    var z1 = Sigmoid(a1);
    var a2 = np.dot(z1, W2) + b2;
    var z2 = Sigmoid(a2);
    var a3 = np.dot(z2, W3) + b3;
    var y = IdentityFunction(a3);

    return y;
}
