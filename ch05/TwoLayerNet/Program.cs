using System.Collections.Generic;
using Numpy;

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

public class TwoLayerNet
{
    public NDarray W1 { get; }

    public NDarray b1 { get; }

    public NDarray W2 { get; }

    public NDarray b2 { get; }

    public IReadOnlyList<ILayer> layers { get; }

    public SoftmaxWithLoss lastLayer { get; }

    public TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std = 0.01)
    {
        // 重みの初期化
        W1 = weight_init_std * np.random.randn(input_size, hidden_size);
        b1 = np.zeros(hidden_size);
        W2 = weight_init_std * np.random.randn(hidden_size, output_size);
        b2 = np.zeros(output_size);

        // レイヤの生成
        layers = new List<ILayer>
        {
            new Affine(W1, b1),
            new Relu(),
            new Affine(W2, b2),
        };
        lastLayer = new SoftmaxWithLoss();
    }

    public NDarray predict(NDarray x)
    {
        foreach (var layer in layers)
        {
            x = layer.forward(x);
        }
        return x;
    }

    // x: 入力データ, t:教師データ
    public NDarray loss(NDarray x, NDarray t)
    {
        var y = predict(x);
        return lastLayer.forward(y, t);
    }

    public NDarray accuracy(NDarray x, NDarray t)
    {
        var y = predict(x);
        y = np.argmax(y, axis: 1);
        if (t.ndim != 1)
        {
            t = np.argmax(t, axis: 1);
        }
        var a = np.sum(np.equal(y, t)) / (double)x.shape[0];
        return a;
    }
}

public interface ILayer
{
    NDarray forward(NDarray x);

    NDarray backward(NDarray dout);
}

public class Relu : ILayer
{
    public NDarray? mask { get; private set; }

    public NDarray forward(NDarray x)
    {
        this.mask = (x <= 0.0);
        var @out = x.copy();
        @out[this.mask] = (NDarray)0;
        return @out;
    }

    public NDarray backward(NDarray dout)
    {
        dout[this.mask] = (NDarray)0;
        var dx = dout;
        return dx;
    }
}

public class Affine : ILayer
{
    public NDarray W { get; }

    public NDarray b { get; }

    public NDarray? x { get; private set; }

    public NDarray? dW { get; private set; }

    public NDarray? db { get; private set; }

    public Affine(NDarray W, NDarray b)
    {
        this.W = W;
        this.b = b;
        x = null;
        dW = null;
        db = null;
    }

    public NDarray forward(NDarray x)
    {
        this.x = x;
        var @out = np.dot(x, W) + b;
        return @out;
    }

    public NDarray backward(NDarray dout)
    {
        var dx = np.dot(dout, W.T);
        dW = np.dot(x!.T, dout);
        db = np.sum(dout, axis: 0);
        return dx;
    }
}

public class SoftmaxWithLoss
{
    /// <summary>損失</summary>
    public NDarray? loss { get; private set; }

    /// <summary>softmaxの出力</summary>
    public NDarray? y { get; private set; }

    /// <summary>教師データ(one-hot vector)</summary>
    public NDarray? t { get; private set; }

    public NDarray forward(NDarray x, NDarray t)
    {
        this.t = t;
        this.y = softmax(x);
        this.loss = cross_entropy_error(this.y, this.t);
        return this.loss;
    }

    public NDarray backward(NDarray dout)
    {
        var batch_size = this.t!.shape[0];
        var dx = (this.y - this.t) / batch_size;
        return dx;
    }

    private static NDarray softmax(NDarray a)
    {
        var c = np.max(a);
        var exp_a = np.exp(a - c);
        var sum_exp_a = np.sum(exp_a);
        var y = exp_a / sum_exp_a;
        return y;
    }

    private static NDarray cross_entropy_error(NDarray y, NDarray t)
    {
        var delta = 1e-7;
        return (-1) * np.sum(t * np.log(y + delta));
    }
}
