using Numpy;

var net = new simpleNet();
Console.WriteLine(net.W); // 重みパラメータ

var x = np.array(0.6, 0.9);
var p = net.predict(x);
Console.WriteLine(p);
Console.WriteLine(np.argmax(p)); // 最大値のインデックス

var t = np.array(0, 0, 1.0); // 正解ラベル
var loss = net.loss(x, t);
Console.WriteLine(loss);

Func<NDarray, NDarray> f = W => net.loss(x, t);
var dW = gradient.numerical_gradient(f, net.W);
Console.WriteLine(dW);

static class functions
{
    public static NDarray softmax(NDarray x)
    {
        x = x - np.max(x, axis: new[] { -1 }, keepdims: true); // オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis: -1, keepdims: true);
    }

    public static NDarray cross_entropy_error(NDarray y, NDarray t)
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
        return (-1) * np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size;
    }
}

static class gradient
{
    static NDarray numerical_gradient_1d(Func<NDarray, NDarray> f, NDarray x)
    {
        var h = 1e-4; // 0.0001
        var grad = np.zeros_like(x);

        for (var idx = 0; idx < x.size; idx++)
        {
            var tmp_val = x[idx];
            x[idx] = tmp_val + h;
            var fxh1 = f(x); // f(x+h)

            x[idx] = tmp_val - h;
            var fxh2 = f(x); // f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h);

            x[idx] = tmp_val; // 値を元に戻す
        }

        return grad;
    }

    public static NDarray numerical_gradient(Func<NDarray, NDarray> f, NDarray X)
    {
        if (X.ndim == 1)
        {
            return numerical_gradient_1d(f, X);
        }
        else
        {
            var grad = np.zeros_like(X);

            for (var idx = 0; idx < X.len; idx++)
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
    public NDarray W { get; }

    public simpleNet()
    {
        //W = np.random.randn(2, 3); // ガウス分布で初期化
        W = np.array(new double[,]
        {
            { 0.47355232, 0.9977393, 0.84668094 },
            { 0.85557411, 0.03563661, 0.69422093 }
        });
    }

    public NDarray predict(NDarray x)
    {
        return np.dot(x, W);
    }

    public NDarray loss(NDarray x, NDarray t)
    {
        var z = predict(x);
        var y = functions.softmax(z);
        var loss = functions.cross_entropy_error(y, t);
        return loss;
    }
}

