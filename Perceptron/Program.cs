using NumSharp;

Console.WriteLine("AND");
Console.WriteLine("x1\tx2\ty");
Console.WriteLine($"0\t0\t{AND(0, 0)}");
Console.WriteLine($"1\t0\t{AND(1, 0)}");
Console.WriteLine($"0\t1\t{AND(0, 1)}");
Console.WriteLine($"1\t1\t{AND(1, 1)}");

Console.WriteLine("OR");
Console.WriteLine("x1\tx2\ty");
Console.WriteLine($"0\t0\t{OR(0, 0)}");
Console.WriteLine($"1\t0\t{OR(1, 0)}");
Console.WriteLine($"0\t1\t{OR(0, 1)}");
Console.WriteLine($"1\t1\t{OR(1, 1)}");

Console.WriteLine("NAND");
Console.WriteLine("x1\tx2\ty");
Console.WriteLine($"0\t0\t{NAND(0, 0)}");
Console.WriteLine($"1\t0\t{NAND(1, 0)}");
Console.WriteLine($"0\t1\t{NAND(0, 1)}");
Console.WriteLine($"1\t1\t{NAND(1, 1)}");

Console.WriteLine("XOR");
Console.WriteLine("x1\tx2\ty");
Console.WriteLine($"0\t0\t{XOR(0, 0)}");
Console.WriteLine($"1\t0\t{XOR(1, 0)}");
Console.WriteLine($"0\t1\t{XOR(0, 1)}");
Console.WriteLine($"1\t1\t{XOR(1, 1)}");

static int AND(int x1, int x2)
{
    var x = np.array<double>(x1, x2);
    var w = np.array(0.5, 0.5);
    var b = -0.7;
    var tmp = np.sum(w * x, typeof(double)) + b;
    if ((double)tmp <= 0)
        return 0;
    else
        return 1;
}

static int NAND(int x1, int x2)
{
    var x = np.array<double>(x1, x2);
    var w = np.array(-0.5, -0.5);
    var b = 0.7;
    var tmp = np.sum(w * x, typeof(double)) + b;
    if ((double)tmp <= 0)
        return 0;
    else
        return 1;
}

static int OR(int x1, int x2)
{
    var x = np.array<double>(x1, x2);
    var w = np.array(0.5, 0.5);
    var b = -0.2;
    var tmp = np.sum(w * x, typeof(double)) + b;
    if ((double)tmp <= 0)
        return 0;
    else
        return 1;
}

static int XOR(int x1, int x2)
{
    var s1 = NAND(x1, x2);
    var s2 = OR(x1, x2);
    var y = AND(s1, s2);
    return y;
}

