using NumSharp;

var init_x = np.array(-3.0, 4.0);
var y = gradient_descent(
    function_2,
    init_x: init_x,
    lr: 0.1,
    step_num: 100);
Console.WriteLine(y.ToString());

// f(x0, x1) = x0 ^ 2 + x1 ^ 2
static double function_2(NDArray x) =>
    x[0] * x[0] + x[1] * x[1];

// 勾配
static NDArray numerical_gradient(
    Func<NDArray, double> f,
    NDArray x)
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

static NDArray gradient_descent(
    Func<NDArray, double> f,
    NDArray init_x,
    double lr = 0.01,
    int step_num = 100)
{
    var x = init_x;

    foreach (var i in Enumerable.Range(0, step_num))
    {
        var grad = numerical_gradient(f, x);
        x -= lr * grad;
    }

    return x;
}
