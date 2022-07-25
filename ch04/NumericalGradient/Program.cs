using NumSharp;

Console.WriteLine("(x0, x1) = (3.0, 4.0) のとき");
Console.WriteLine(
    numerical_gradient(
        function_2,
        np.array(3.0, 4.0)).ToString());

Console.WriteLine("(x0, x1) = (0.0, 2.0) のとき");
Console.WriteLine(
    numerical_gradient(
        function_2,
        np.array(0.0, 2.0)).ToString());

Console.WriteLine("(x0, x1) = (3.0, 0.0) のとき");
Console.WriteLine(
    numerical_gradient(
        function_2,
        np.array(3.0, 0.0)).ToString());

Console.ReadLine();

static double function_2(NDArray x) =>
    x[0] * x[0] + x[1] * x[1];

static NDArray numerical_gradient(Func<NDArray, double> f, NDArray x)
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
