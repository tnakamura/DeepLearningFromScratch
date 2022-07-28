using NumSharp;

var net = new simpleNet();
Console.WriteLine(net.W.ToString()); // 重みパラメータ

var x = np.array(0.6, 0.9);
var p = net.predict(x);
Console.WriteLine(p.ToString());
Console.WriteLine(np.argmax(p)); // 最大値のインデックス

var t = np.array(0, 0, 1.0); // 正解ラベル
var loss = net.loss(x, t);
Console.WriteLine(loss.ToString());

Func<NDArray, NDArray> f = W => net.loss(x, t);
var dW = gradient.numerical_gradient(f, net.W);
Console.WriteLine(dW.ToString());

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

class simpleNet
{
    public NDArray W { get; }

    public simpleNet()
    {
        // ゼロから作る Deep Learning と同じ重みパラメーターを指定する。
        W = np.array(new double[,]
        {
            { 0.47355232, 0.9977393, 0.84668094 },
            { 0.85557411, 0.03563661, 0.69422093 }
        });
    }

    public NDArray predict(NDArray x)
    {
        // NumSharp が 1-D と 2-D の内積をサポートしていなかったので、
        // x を 2-D に変換して計算したあと、1-D に戻して回避。
        // 汎用的な手法ではない。
        x = x.reshape_unsafe(2, 2);
        var y = np.dot(x, W);
        return y.reshape_unsafe(3);
    }

    public NDArray loss(NDArray x, NDArray t)
    {
        var z = predict(x);
        var y = functions.softmax(z);
        var loss = functions.cross_entropy_error(y, t);
        return loss;
    }
}

