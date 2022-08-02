using NumSharp;
using static functions;

var net = new TwoLayerNet(input_size: 784, hidden_size: 100, output_size: 10);
Console.WriteLine(net.W1.Shape);
Console.WriteLine(net.b1.Shape);
Console.WriteLine(net.W2.Shape);
Console.WriteLine(net.b2.Shape);

{
    var x = np.random.rand(100, 784); // ダミーの入力データ(100枚分)
    var y = net.predict(x);
    Console.WriteLine(y.ToString());
}

{
    var x = np.random.rand(100, 784); // ダミーの入力データ(100枚分)
    var t = np.random.rand(100, 10); // ダミーの正解ラベル

    var grads = net.numerical_gradient(x, t); // 勾配を計算

    Console.WriteLine(grads.W1.Shape);
    Console.WriteLine(grads.b1.Shape);
    Console.WriteLine(grads.W2.Shape);
    Console.WriteLine(grads.b2.Shape);
}


Console.ReadLine();

public class TwoLayerNet
{
    public NDArray W1 { get; }

    public NDArray W2 { get; }

    public NDArray b1 { get; }

    public NDArray b2 { get; }

    public TwoLayerNet(
        int input_size,
        int hidden_size,
        int output_size,
        double weight_init_std = 0.01)
    {
        W1 = weight_init_std * np.random.randn(input_size, hidden_size);
        b1 = np.zeros(hidden_size);
        W2 = weight_init_std * np.random.randn(hidden_size, output_size);
        b2 = np.zeros(output_size);
    }

    public NDArray predict(NDArray x)
    {
        var a1 = np.dot(x, W1) + b1;
        var z1 = sigmoid(a1);
        var a2 = np.dot(z1, W2) + b2;
        var y = softmax(a2);
        return y;
    }

    public NDArray loss(NDArray x, NDArray t)
    {
        var y = predict(x);
        return cross_entropy_error(y, t);
    }

    public NDArray accuracy(NDArray x, NDArray t)
    {
        var y = predict(x);
        y = np.argmax(y, axis: 1);
        t = np.argmax(t, axis: 1);
        var accuracy = np.sum(y == t) / (float)x.shape[0];
        return accuracy;
    }

    // x:入力データ, t:教師データ
    public (NDArray W1, NDArray b1, NDArray W2, NDArray b2) numerical_gradient(NDArray x, NDArray t)
    {
        NDArray loss_W(NDArray W) => loss(x, t);
        var W1 = gradient.numerical_gradient(loss_W, this.W1);
        var b1 = gradient.numerical_gradient(loss_W, this.b1);
        var W2 = gradient.numerical_gradient(loss_W, this.W2);
        var b2 = gradient.numerical_gradient(loss_W, this.b2);
        return (W1, b1, W2, b2);
    }
}

static class functions
{
    public static NDArray softmax(NDArray x)
    {
        var c = np.max(x);
        var exp_x = np.exp(x - c);
        var sum_exp_x = np.sum(exp_x, NPTypeCode.Double);
        var y = exp_x / sum_exp_x;
        return y;
    }

    public static NDArray sigmoid(NDArray x)
    {
        return 1 / (1 + np.exp(x * -1));
    }

    public static NDArray cross_entropy_error(NDArray y, NDArray t)
    {
        var delta = 1e-7;
        return (-1) * np.sum(t * np.log(y + delta), NPTypeCode.Double);
    }
}

static class gradient
{
    static NDArray numerical_gradient_1d(Func<NDArray, NDArray> f, NDArray x)
    {
        var h = 1e-4; // 0.0001
        var grad = np.zeros_like(x); // x と同じ形状の配列を生成

        foreach (var idx in Enumerable.Range(0, x.size))
        {
            var tmp_val = (double)x[idx];

            // f(x + h) の計算
            x[idx] = tmp_val + h;
            var fxh1 = f(x);

            // f(x - h) の計算
            x[idx] = tmp_val - h;
            var fxh2 = f(x);

            grad[idx] = (fxh1 - fxh2) / (2 * h);
            x[idx] = tmp_val; // 値を元に戻す
        }

        return grad;
    }

    public static NDArray numerical_gradient(Func<NDArray, NDArray> f, NDArray X)
    {
        if (X.ndim == 1)
        {
            return numerical_gradient_1d(f, X);
        }
        else
        {
            var grad = np.zeros_like(X);

            for (var idx = 0; idx < X.ndim; idx++)
            {
                var x = X[idx];
                grad[idx] = numerical_gradient_1d(f, x);
            }

            return grad;
        }
    }
}
