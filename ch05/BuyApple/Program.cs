using NumSharp;

NDArray apple = 100;
NDArray apple_num = 2;
NDArray tax = 1.1;

var mul_apple_layer = new MulLayer();
var mul_tax_layer = new MulLayer();

// forward
var apple_price = mul_apple_layer.forward(apple, apple_num);
var price = mul_tax_layer.forward(apple_price, tax);
Console.WriteLine(price.ToString());

// backward
NDArray dprice = 1;
var (dapple_price, dtax) = mul_tax_layer.backward(dprice);
var (dapple, dapple_num) = mul_apple_layer.backward(dapple_price);
Console.WriteLine($"{dapple} {dapple_num} {dtax}");

class MulLayer
{
    NDArray? x;
    NDArray? y;

    public NDArray forward(NDArray x, NDArray y)
    {
        this.x = x;
        this.y = y;
        var @out = x * y;
        return @out;
    }

    public (NDArray dx, NDArray dy) backward(NDArray dout)
    {
        var dx = dout * this.y;
        var dy = dout * this.x;
        return (dx, dy);
    }
}

