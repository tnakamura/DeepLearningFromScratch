using System.IO.Compression;
using NumSharp;

await Mnist.DownloadAsync();

Console.ReadLine();

class Mnist
{
    const string UrlBase = "http://yann.lecun.com/exdb/mnist/";

    const int ImageSize = 784;

    static readonly IReadOnlyDictionary<string, string> s_keyFile = new Dictionary<string, string>
    {
        ["train_img"] = "train-images-idx3-ubyte.gz",
        ["train_label"] = "train-labels-idx1-ubyte.gz",
        ["test_img"] = "t10k-images-idx3-ubyte.gz",
        ["test_label"] = "t10k-labels-idx1-ubyte.gz",
    };

    static readonly string s_datasetDir = AppContext.BaseDirectory;

    static readonly HttpClient s_httpClient = new HttpClient();

    public static async Task DownloadAsync()
    {
        foreach (var v in s_keyFile.Values)
        {
            await DownloadAsync(v);
        }
    }

    private static async Task DownloadAsync(string fileName)
    {
        var filePath = Path.Combine(s_datasetDir, fileName);
        if (File.Exists(filePath))
        {
            return;
        }
        Console.WriteLine("Downloading " + fileName + " ... ");

        var response = await s_httpClient.GetAsync(UrlBase + fileName);
        using var stream = await response.Content.ReadAsStreamAsync();
        using var f = File.OpenWrite(filePath);
        await stream.CopyToAsync(f);
        Console.WriteLine("Done");
    }

    private static async Task<NDArray> LoadLabelsAsync(string fileName)
    {
        var filePath = Path.Combine(s_datasetDir, fileName);

        Console.WriteLine("Converting " + fileName + " to NumSharp Array ...");

        using var f = File.OpenRead(filePath);
        using var gzip = new GZipStream(f, CompressionLevel.Fastest);
        var buffer = new byte[f.Length - 8];
        await gzip.ReadAsync(buffer, 8, buffer.Length);
        var labels = np.frombuffer(buffer.ToArray(), np.uint8);
        Console.WriteLine("Done");

        return labels;
    }

    private static async Task<NDArray> LoadImagesAsync(string fileName)
    {
        var filePath = Path.Combine(s_datasetDir, fileName);

        Console.WriteLine("Converting " + fileName + " to NumSharp Array ...");

        using var f = File.OpenRead(filePath);
        using var gzip = new GZipStream(f, CompressionLevel.Fastest);
        var buffer = new byte[f.Length - 8];
        await gzip.ReadAsync(buffer, 8, buffer.Length);
        var data = np.frombuffer(buffer.ToArray(), np.uint8);
        data = data.reshape(-1, ImageSize);
        Console.WriteLine("Done");

        return data;
    }

    public void Init()
    {

    }
}
