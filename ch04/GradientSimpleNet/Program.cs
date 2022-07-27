using NumSharp;
using static functions;
using static gradient;

var net = new simpleNet();
Console.WriteLine(net.W.ToString());

var x = np.array(0.6, 0.9);
var p = net.predict(x);
Console.WriteLine("予測");
Console.WriteLine(p.ToString());

var t = np.array(0, 0, 1.0);
var loss = net.loss(x, t);
Console.WriteLine("損失関数の値");
Console.WriteLine(loss.ToString());

Func<NDArray, double> f = W => net.loss(x, t);
var dW = numerical_gradient_2d(f, net.W);
Console.WriteLine("勾配");
Console.WriteLine(dW.ToString());

static class functions
{
    public static NDArray softmax(NDArray x)
    {
        x = x - np.max(x, axis: -1, keepdims: true); // オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis: -1, keepdims: true, typeCode: NPTypeCode.Double);
    }

    public static NDArray cross_entropy_error(NDArray y, NDArray t)
    {
        if (y.ndim == 1)
        {
            t = t.reshape(1, t.size);
            y = y.reshape(1, y.size);
        }

        // 教師データが one-hot-vector の場合、正解ラベルのインデックスに変換
        if (t.size == y.size)
        {
            t = t.argmax(1);
        }

        var batch_size = y.shape[0];
        return (-1) * np.sum(np.log(y[np.arange(batch_size), t] + 1e-7), NPTypeCode.Double) / batch_size;
    }
}

static class gradient
{
    static NDArray numerical_gradient_1d(Func<NDArray, double> f, NDArray x)
    {
        var h = 1e-4; // 0.0001
        var grad = np.zeros_like(x);

        foreach (var idx in Enumerable.Range(0, x.size))
        {
            var tmp_val = (double)x[idx];
            x[idx] = tmp_val + h;
            var fxh1 = f(x); // f(x+h)

            x[idx] = tmp_val - h;
            var fxh2 = f(x); // f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h);

            x[idx] = tmp_val; // 値を元に戻す
        }

        return grad;
    }

    public static NDArray numerical_gradient_2d(Func<NDArray, double> f, NDArray X)
    {
        if (X.ndim == 1)
        {
            return numerical_gradient_1d(f, X);
        }
        else
        {
            var grad = np.zeros_like(X);

            for (var idx = 0; idx < X.size; idx++)
            {
                var x = X[idx];
                grad[idx] = numerical_gradient_1d(f, x);
            }

            return grad;
        }
    }
}

class simpleNet
{
    public NDArray W { get; }

    public simpleNet()
    {
        //W = np.random.randn(2, 3); // ガウス分布で初期化
        W = np.array(new double[,]
        {
            { 0.47355232, 0.9977393, 0.84668094 },
            { 0.85557411, 0.03563661, 0.69422093 }
        });
    }

    public NDArray predict(NDArray x)
    {
        return np.dot(x, W);
    }

    public NDArray loss(NDArray x, NDArray t)
    {
        var z = predict(x);
        var y = softmax(z);
        var loss = cross_entropy_error(y, t);
        return loss;
    }
}

