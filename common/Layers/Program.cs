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

