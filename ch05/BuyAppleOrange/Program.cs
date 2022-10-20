using NumSharp;

NDArray apple = 100;
NDArray apple_num = 2;
NDArray orange = 150;
NDArray orange_num = 3;
NDArray tax = 1.1;

// layer
var mul_apple_layer = new MulLayer();
var mul_orange_layer = new MulLayer();
var add_apple_orange_layer = new AddLayer();
var mul_tax_layer = new MulLayer();

// forward
var apple_price = mul_apple_layer.forward(apple, apple_num);
var orange_price = mul_orange_layer.forward(orange, orange_num);
var all_price = add_apple_orange_layer.forward(apple_price, orange_price);
var price = mul_tax_layer.forward(all_price, tax);

// backward
NDArray dprice = 1;
var (dall_price, dtax) = mul_tax_layer.backward(dprice);
var (dapple_price, dorange_price) = add_apple_orange_layer.backward(dall_price);
var (dorange, dorange_num) = mul_orange_layer.backward(dorange_price);
var (dapple, dapple_num) = mul_apple_layer.backward(dapple_price);

Console.WriteLine(price.ToString());
Console.WriteLine($"{dapple_num} {dapple} {dorange} {dorange_num} {dtax}");

class AddLayer
{
    public NDArray forward(NDArray x, NDArray y)
    {
        var @out = x + y;
        return @out;
    }

    public (NDArray dx, NDArray dy) backward(NDArray dout)
    {
        var dx = dout * 1;
        var dy = dout * 1;
        return (dx, dy);
    }
}

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

