using System.IO.Compression;
using NumSharp;

namespace MnistSharp;

public static class Mnist
{
    private const string UrlBase = "http://yann.lecun.com/exdb/mnist/";

    private const int ImageSize = 784;

    private static readonly Dictionary<string, string> s_keyFile = new Dictionary<string, string>
    {
        [nameof(Dataset.TrainImage)] = "train-images-idx3-ubyte.gz",
        [nameof(Dataset.TrainLabel)] = "train-labels-idx1-ubyte.gz",
        [nameof(Dataset.TestImage)] = "t10k-images-idx3-ubyte.gz",
        [nameof(Dataset.TestLabel)] = "t10k-labels-idx1-ubyte.gz",
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
        var trainImage = await LoadImagesAsync(s_keyFile[nameof(Dataset.TrainImage)]);
        var trainLabel = await LoadLabelsAsync(s_keyFile[nameof(Dataset.TrainLabel)]);
        var testImage = await LoadImagesAsync(s_keyFile[nameof(Dataset.TestImage)]);
        var testLabel = await LoadLabelsAsync(s_keyFile[nameof(Dataset.TestLabel)]);
        return new Dataset(trainImage, trainLabel, testImage, testLabel);
    }

    public static async Task InitializeAsync()
    {
        await DownloadAsync();
        var dataset = await ConvertNumSharpAsync();
        Console.WriteLine("Creating npy files ...");
        np.save(Path.Combine(s_datasetDir, nameof(dataset.TrainImage)), dataset.TrainImage);
        np.save(Path.Combine(s_datasetDir, nameof(dataset.TrainLabel)), dataset.TrainLabel);
        np.save(Path.Combine(s_datasetDir, nameof(dataset.TestImage)), dataset.TestImage);
        np.save(Path.Combine(s_datasetDir, nameof(dataset.TestLabel)), dataset.TestLabel);
        Console.WriteLine("Done!");
    }

    private static NDArray ChangeOneHotLabel(NDArray x)
    {
        var t = np.zeros(x.size, 10);
        var i = 0;
        foreach (var n in x)
        {
            var j = Convert.ToInt32(n);
            t[i, j] = 1;
            i++;
        }
        Console.WriteLine(t.ToString());
        return t;
    }

    public static async Task<Dataset> LoadAsync(bool normalize = true, bool flatten = true, bool oneHotLabel = false)
    {
        if (!File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.TrainImage) + ".npy")) ||
            !File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.TrainLabel) + ".npy")) ||
            !File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.TestImage) + ".npy")) ||
            !File.Exists(Path.Combine(s_datasetDir, nameof(Dataset.TestLabel) + ".npy")))
        {
            await InitializeAsync();
        }

        var trainImage = np.load(Path.Combine(s_datasetDir, nameof(Dataset.TrainImage) + ".npy"));
        var trainLabel = np.load(Path.Combine(s_datasetDir, nameof(Dataset.TrainLabel) + ".npy"));
        var testImage = np.load(Path.Combine(s_datasetDir, nameof(Dataset.TestImage) + ".npy"));
        var testLabel = np.load(Path.Combine(s_datasetDir, nameof(Dataset.TestLabel) + ".npy"));

        if (normalize)
        {
            trainImage = trainImage.astype(np.float32);
            trainImage /= 255.0;
            testImage = testImage.astype(np.float32);
            testImage /= 255.0;
        }

        if (oneHotLabel)
        {
            trainLabel = ChangeOneHotLabel(trainLabel);
            testLabel = ChangeOneHotLabel(testLabel);
        }

        if (!flatten)
        {
            trainImage = trainImage.reshape(-1, 1, 28, 28);
            testImage = testImage.reshape(-1, 1, 28, 28);
        }

        return new Dataset(trainImage, trainLabel, testImage, testLabel);
    }
}

public record Dataset(
    NDArray TrainImage,
    NDArray TrainLabel,
    NDArray TestImage,
    NDArray TestLabel);

