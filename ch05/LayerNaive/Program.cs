var apple = 100m;
var apple_num = 2m;
var tax = 1.1m;

var mul_apple_layer = new MulLayer();
var mul_tax_layer = new MulLayer();

// forward
var apple_price = mul_apple_layer.forward(apple, apple_num);
var price = mul_tax_layer.forward(apple_price, tax);
Console.WriteLine(price);

// backward
var dprice = 1m;
var (dapple_price, dtax) = mul_tax_layer.backward(dprice);
var (dapple, dapple_num) = mul_apple_layer.backward(dapple_price);
Console.WriteLine($"{dapple} {dapple_num} {dtax}");

class MulLayer
{
    decimal x;
    decimal y;

    public decimal forward(decimal x, decimal y)
    {
        this.x = x;
        this.y = y;
        var @out = x * y;
        return @out;
    }

    public (decimal dx, decimal dy) backward(decimal dout)
    {
        var dx = dout * this.y;
        var dy = dout * this.x;
        return (dx, dy);
    }
}
