using NumSharp;

{
    // 正解を 2 とする
    var t = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };

    // 例1: 「2」の確率が最も高い場合 (0.6)
    var y = new double[] { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };
    Console.WriteLine(mean_squared_error(np.array(y), np.array(t)).ToString());

    // 例2: 「7」の確率が最も高い場合 (0.6)
    y = new double[] { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0 };
    Console.WriteLine(mean_squared_error(np.array(y), np.array(t)).ToString());
}

{
    var t = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    var y = new double[] { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };
    Console.WriteLine(cross_entropy_error(np.array(y), np.array(t)).ToString());

    y = new double[] { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0 };
    Console.WriteLine(cross_entropy_error(np.array(y), np.array(t)).ToString());
}

Console.ReadLine();

// 2乗和誤差
static NDArray mean_squared_error(NDArray y, NDArray t)
{
    return 0.5 * np.sum((y - t) * (y - t), NPTypeCode.Double);
}

// 交差エントロピー誤差
static NDArray cross_entropy_error(NDArray y, NDArray t)
{
    var delta = 1e-7;
    return (-1) * np.sum(t * np.log(y + delta), NPTypeCode.Double);
}
