using Numpy;

var x = np.array(new double[,]
{
    { 1.0, -0.5 },
    { -2.0, 3.0 },
});
//var x = np.array(new[] { new[] { 1.0, -0.5 }, new[] { -2.0, 3.0 } });
Console.WriteLine(x);

// <= が実装されていなかったので
// (x <= 0.0) = !(x > 0.0) で代用
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
