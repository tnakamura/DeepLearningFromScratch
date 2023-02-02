using Numpy;
/*
{
    var x = np.array(new double[,]
    {
    { 1.0, -0.5 },
    { -2.0, 3.0 },
    });
    Console.WriteLine(x);

    var mask = (x <= 0.0);
    Console.WriteLine(mask);

    var relu = new Relu();
    var @out = relu.forward(x);
    Console.WriteLine(relu.mask);
    Console.WriteLine(@out);

    var dx = relu.backward(@out);
    Console.WriteLine(dx);
}
*/

{
    var X_dot_W = np.array(new double[,]
    {
        { 0, 0, 0, },
        { 10, 10, 10 }
    });
    var B = np.array(new double[] { 1, 2, 3 });
    Console.WriteLine(X_dot_W);
    Console.WriteLine(X_dot_W + B);

    var dY = np.array(new double[,]
    {
        { 1, 2, 3 },
        { 4, 5, 6 }
    });
    Console.WriteLine(dY);

    var dB = np.sum(dY, axis: 0);
    Console.WriteLine(dB);

    var X = np.array(new double[] { 0, 0, 0 });
    var W = np.array(new double[] { 10, 10, 10 });
    var b = np.array(new double[] { 1, 2, 3 });
    var affine = new Affine(W, b);
    var X_dot_W_plus_B = affine.forward(X);
    Console.WriteLine(X_dot_W_plus_B);
}

public class Affine
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

public class Sigmoid
{
    private NDarray? _out;

    public NDarray forward(NDarray x)
    {
        var @out = 1 / (1 + np.exp(-x));
        _out = @out;

        return @out;
    }

    public NDarray backward(NDarray dout)
    {
        var dx = dout * (1.0 - _out) * _out;

        return dx;
    }
}


public class Relu
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

