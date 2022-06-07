using System.IO.Compression;
using NumSharp;

var ds = await Mnist.LoadAsync(oneHotLabel: true);

Console.ReadLine();

static class Mnist
{
    private const string UrlBase = "http://yann.lecun.com/exdb/mnist/";

    private const int ImageSize = 784;

    private static readonly Dictionary<string, string> s_keyFile = new Dictionary<string, string>
    {
        ["train_img"] = "train-images-idx3-ubyte.gz",
        ["train_label"] = "train-labels-idx1-ubyte.gz",
        ["test_img"] = "t10k-images-idx3-ubyte.gz",
        ["test_label"] = "t10k-labels-idx1-ubyte.gz",
    };

    private static readonly string s_datasetDir = AppContext.BaseDirectory;

    private static readonly HttpClient s_httpClient = new HttpClient();

    private static async Task DownloadAsync()
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
        using var gzip = new GZipStream(f, CompressionMode.Decompress);
        using var memory = new MemoryStream();
        await gzip.CopyToAsync(memory);

        var buffer = memory.ToArray().AsSpan(8).ToArray();
        var labels = np.frombuffer(buffer, np.uint8);
        Console.WriteLine("Done");

        return labels;
    }

    private static async Task<NDArray> LoadImagesAsync(string fileName)
    {
        var filePath = Path.Combine(s_datasetDir, fileName);

        Console.WriteLine("Converting " + fileName + " to NumSharp Array ...");

        using var f = File.OpenRead(filePath);
        using var gzip = new GZipStream(f, CompressionMode.Decompress);
        using var memory = new MemoryStream();
        await gzip.CopyToAsync(memory);

        var buffer = memory.ToArray().AsSpan(16).ToArray();
        var data = np.frombuffer(buffer, np.uint8);
        data = data.reshape(-1, ImageSize);
        Console.WriteLine("Done");

        return data;
    }

    private static async Task<Dataset> ConvertNumSharpAsync()
    {
        var dataset = new Dataset();
        dataset.train_img = await LoadImagesAsync(s_keyFile["train_img"]);
        dataset.train_label = await LoadLabelsAsync(s_keyFile["train_label"]);
        dataset.test_img = await LoadImagesAsync(s_keyFile["test_img"]);
        dataset.test_label = await LoadLabelsAsync(s_keyFile["test_label"]);
        return dataset;
    }

    public static async Task InitializeAsync()
    {
        await DownloadAsync();
        var dataset = await ConvertNumSharpAsync();
        Console.WriteLine("Creating npy files ...");
        np.save(Path.Combine(s_datasetDir, nameof(dataset.train_img)), dataset.train_img);
        np.save(Path.Combine(s_datasetDir, nameof(dataset.train_label)), dataset.train_label);
        np.save(Path.Combine(s_datasetDir, nameof(dataset.test_img)), dataset.test_img);
        np.save(Path.Combine(s_datasetDir, nameof(dataset.test_label)), dataset.test_label);
        Console.WriteLine("Done!");
    }

    private static NDArray ChangeOneHotLabel(NDArray x)
    {
        var t = np.zeros(x.size, 10);
        for (var idx = 0; idx < x.size; idx++)
        {
            var row = t[idx];
            row[x[idx]] = 1;
        }
        return t;
    }

    public static async Task<Dataset> LoadAsync(bool normalize = true, bool flatten = true, bool oneHotLabel = false)
    {
        if (!File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.train_img) + ".npy")) ||
            !File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.train_label) + ".npy")) ||
            !File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.test_img) + ".npy")) ||
            !File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.test_label) + ".npy")))
        {
            await InitializeAsync();
        }

        var dataset = new Dataset();
        dataset.train_img = np.load(Path.Combine(s_datasetDir, nameof(Dataset.train_img) + ".npy"));
        dataset.train_label = np.load(Path.Combine(s_datasetDir, nameof(Dataset.train_label) + ".npy"));
        dataset.test_img = np.load(Path.Combine(s_datasetDir, nameof(Dataset.test_img) + ".npy"));
        dataset.test_label = np.load(Path.Combine(s_datasetDir, nameof(Dataset.test_label) + ".npy"));

        if (normalize)
        {
            dataset.train_img = dataset.train_img.astype(np.float32);
            dataset.train_img /= 255.0;
            dataset.test_img = dataset.test_img.astype(np.float32);
            dataset.test_img /= 255.0;
        }

        if (oneHotLabel)
        {
            dataset.train_label = ChangeOneHotLabel(dataset.train_label);
            dataset.test_label = ChangeOneHotLabel(dataset.test_label);
        }

        if (!flatten)
        {
            dataset.train_img = dataset.train_img.reshape(-1, 1, 28, 28);
            dataset.test_img = dataset.test_img.reshape(-1, 1, 28, 28);
        }

        return dataset;
    }
}
record struct Dataset(NDArray train_img, NDArray train_label, NDArray test_img, NDArray test_label);

