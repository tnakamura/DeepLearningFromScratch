using Numpy;

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
