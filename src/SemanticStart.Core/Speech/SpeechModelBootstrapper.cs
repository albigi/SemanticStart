using System.Diagnostics;
using System.Net;
using SemanticStart.Core;

namespace SemanticStart.Core.Speech;

/// <summary>
/// Fetches the speech models on first use, the way <c>EmbeddingModelBootstrapper</c> fetches the
/// embedding model: into the same models directory, with the same resumable-by-restart temp file,
/// the same size floor against a truncated download, and the same progress reporting so the
/// settings page can show one bar.
///
/// <para>
/// No model is committed to the repository. A speech model is tens of megabytes of binary that
/// every clone would carry forever, and the release zip would grow by the same amount for users
/// who never dictate. Downloading it also keeps the decision where it belongs: dictation is opt-in
/// and the download is what the user is consenting to.
/// </para>
///
/// <para>
/// <b>What is downloaded, and under what terms.</b>
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///     <c>sherpa-onnx-streaming-zipformer-en-2023-06-26</c> (Apache-2.0), the int8-quantised
///     encoder/decoder/joiner and their token table: about 73 MB in total, which is the same order
///     as the 90 MB embedding model already downloaded on first run. Trained on LibriSpeech by the
///     Next-gen Kaldi project; the float32 encoder alone is 262 MB, which is the reason the
///     quantised files are the ones named here.
///     </description>
///   </item>
///   <item>
///     <description>
///     <c>silero_vad.onnx</c> v5.1 (MIT), 2.2 MB, pinned to a tag rather than a branch so the file
///     cannot change underneath a released build. Its SHA-256 is verified because it is small
///     enough that hashing costs nothing and it is fetched from a raw file URL rather than a
///     release asset.
///     </description>
///   </item>
/// </list>
/// <para>
/// The Zipformer files are checked by size rather than by hash: the hashes are not published
/// alongside the model, and a hash recorded from one download here would be a claim about that
/// download rather than about the model. Both are served over HTTPS from the projects' own
/// hosting.
/// </para>
/// </summary>
public sealed class SpeechModelBootstrapper
{
    public const string ModelId = "sherpa-onnx-streaming-zipformer-en-2023-06-26";

    private const string ModelBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/";

    public const string DefaultEncoderUrl = ModelBaseUrl + "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
    public const string DefaultDecoderUrl = ModelBaseUrl + "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
    public const string DefaultJoinerUrl = ModelBaseUrl + "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
    public const string DefaultTokensUrl = ModelBaseUrl + "tokens.txt";

    public const string DefaultVadUrl =
        "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/src/silero_vad/data/silero_vad.onnx";

    /// <summary>SHA-256 of silero_vad.onnx at tag v5.1.</summary>
    public const string VadSha256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f";

    /// <summary>What the user is told they are about to download, before it starts.</summary>
    public const long ApproximateDownloadBytes = 75L * 1024 * 1024;

    private const long MinimumEncoderBytes = 50_000_000;
    private const long MinimumDecoderBytes = 500_000;
    private const long MinimumJoinerBytes = 100_000;
    private const long MinimumTokensBytes = 1_000;
    private const long MinimumVadBytes = 1_000_000;

    /// <summary>
    /// Shared by every bootstrapper that is not handed a client of its own. The settings window,
    /// the warm start, and the harness each construct one of these, and a per-instance client
    /// would leak a connection pool every time; a download this size also wants a long timeout,
    /// which is the only reason the default client is not used.
    /// </summary>
    private static readonly HttpClient SharedHttpClient = new() { Timeout = TimeSpan.FromMinutes(10) };

    private readonly HttpClient _httpClient;
    private readonly string _modelsDirectory;

    public SpeechModelBootstrapper(HttpClient? httpClient = null, string? modelsDirectory = null)
    {
        _httpClient = httpClient ?? SharedHttpClient;
        _modelsDirectory = modelsDirectory ?? AppPaths.ModelsDirectory;
    }

    public string EncoderPath => Path.Combine(_modelsDirectory, ModelId + "-encoder.int8.onnx");

    public string DecoderPath => Path.Combine(_modelsDirectory, ModelId + "-decoder.int8.onnx");

    public string JoinerPath => Path.Combine(_modelsDirectory, ModelId + "-joiner.int8.onnx");

    public string TokensPath => Path.Combine(_modelsDirectory, ModelId + "-tokens.txt");

    public string VadPath => Path.Combine(_modelsDirectory, "silero_vad.onnx");

    /// <summary>
    /// Whether every file is already on disk, so dictation can start without a download. Read at
    /// startup to decide whether loading the engine needs the user's consent first.
    /// </summary>
    public bool IsDownloaded =>
        IsUsableFile(EncoderPath, MinimumEncoderBytes)
        && IsUsableFile(DecoderPath, MinimumDecoderBytes)
        && IsUsableFile(JoinerPath, MinimumJoinerBytes)
        && IsUsableFile(TokensPath, MinimumTokensBytes)
        && IsUsableFile(VadPath, MinimumVadBytes);

    public async Task<SpeechModelFiles> EnsureAsync(
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        AppPaths.EnsureCreated();
        Directory.CreateDirectory(_modelsDirectory);

        using var activity = SpeechDiagnostics.StartActivity(SpeechDiagnostics.ModelEnsureActivity);
        activity?.SetTag("speech.model", ModelId);
        activity?.SetTag("speech.model.already_present", IsDownloaded);

        // Weighted by size so the bar moves at roughly a constant rate: the encoder is the
        // download, and the other three together are rounding error.
        await DownloadIfNeededAsync(DefaultEncoderUrl, EncoderPath, MinimumEncoderBytes, null, ScaleProgress(progress, 0.0, 0.92), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(DefaultDecoderUrl, DecoderPath, MinimumDecoderBytes, null, ScaleProgress(progress, 0.92, 0.94), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(DefaultJoinerUrl, JoinerPath, MinimumJoinerBytes, null, ScaleProgress(progress, 0.94, 0.95), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(DefaultTokensUrl, TokensPath, MinimumTokensBytes, null, ScaleProgress(progress, 0.95, 0.96), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(DefaultVadUrl, VadPath, MinimumVadBytes, VadSha256, ScaleProgress(progress, 0.96, 1.0), cancellationToken).ConfigureAwait(false);
        progress?.Report(1.0);

        return new SpeechModelFiles(EncoderPath, DecoderPath, JoinerPath, TokensPath, VadPath);
    }

    private static IProgress<double>? ScaleProgress(IProgress<double>? progress, double start, double end)
    {
        return progress is null
            ? null
            : new Progress<double>(value => progress.Report(start + ((end - start) * Math.Clamp(value, 0.0, 1.0))));
    }

    private async Task DownloadIfNeededAsync(
        string url,
        string path,
        long minimumBytes,
        string? expectedSha256,
        IProgress<double>? progress,
        CancellationToken cancellationToken)
    {
        if (IsUsableFile(path, minimumBytes))
        {
            progress?.Report(1.0);
            return;
        }

        // Per file rather than per model: a download that stalls stalls on one of five files, and
        // the span says which, how big it was and how long it took without anyone adding logging
        // to a loop that runs a thousand times a megabyte.
        using var activity = SpeechDiagnostics.StartActivity(SpeechDiagnostics.ModelDownloadActivity);
        activity?.SetTag("http.url", url);
        activity?.SetTag("speech.model.file", Path.GetFileName(path));

        SpeechDiagnostics.Report("model.download", $"Downloading {Path.GetFileName(path)} from {url}.");

        var tempPath = path + ".tmp";
        try
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);

            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            using HttpResponseMessage response = await _httpClient
                .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            if (response.StatusCode == HttpStatusCode.NotFound)
                throw new SpeechModelDownloadException($"Speech model file was not found at {url}.");

            response.EnsureSuccessStatusCode();

            var contentLength = response.Content.Headers.ContentLength;
            await using Stream source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            await using (var destination = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1024 * 1024, useAsync: true))
            {
                var buffer = new byte[1024 * 1024];
                long totalRead = 0;
                int read;
                while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                {
                    await destination.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                    totalRead += read;
                    if (contentLength is > 0)
                        progress?.Report((double)totalRead / contentLength.Value);
                }

                activity?.SetTag("http.response.body.size", totalRead);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            activity?.SetStatus(ActivityStatusCode.Error, "cancelled");
            TryDelete(tempPath);
            throw;
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or OperationCanceledException or InvalidOperationException)
        {
            activity?.SetStatus(ActivityStatusCode.Error, ex.GetType().Name);
            TryDelete(tempPath);
            throw new SpeechModelDownloadException(
                $"Could not download the speech model from {url}. Check network connectivity or pre-place the file at {path}.",
                ex);
        }

        if (!IsUsableFile(tempPath, minimumBytes))
        {
            TryDelete(tempPath);
            throw new SpeechModelDownloadException(
                $"The speech model file downloaded from {url} was too small or empty. Try again: {path}");
        }

        if (expectedSha256 is not null && !MatchesSha256(tempPath, expectedSha256))
        {
            TryDelete(tempPath);
            throw new SpeechModelDownloadException(
                $"The file downloaded from {url} did not match its expected checksum and was discarded.");
        }

        File.Move(tempPath, path, overwrite: true);
        progress?.Report(1.0);
    }

    internal static bool MatchesSha256(string path, string expected)
    {
        using var stream = File.OpenRead(path);
        var actual = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));
        return string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsUsableFile(string path, long minimumBytes) =>
        File.Exists(path) && new FileInfo(path).Length >= minimumBytes;

    /// <summary>
    /// Removes a half-written temp file. Failing to is not worth failing the download over - the
    /// next attempt overwrites it - but it is reported, because a temp file that cannot be deleted
    /// is usually a permissions or antivirus problem that will fail the retry as well.
    /// </summary>
    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            SpeechDiagnostics.ReportFailure(
                "model.cleanup",
                $"Could not delete the partial download at {path}.",
                ex);
        }
    }
}

public sealed record SpeechModelFiles(
    string EncoderPath,
    string DecoderPath,
    string JoinerPath,
    string TokensPath,
    string VadPath);

public sealed class SpeechModelDownloadException : Exception
{
    public SpeechModelDownloadException(string message)
        : base(message)
    {
    }

    public SpeechModelDownloadException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
