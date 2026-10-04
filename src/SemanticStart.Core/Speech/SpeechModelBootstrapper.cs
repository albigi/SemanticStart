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
    /// <summary>
    /// What the user is told they are about to download, before it starts. Static because the
    /// settings page asks before any bootstrapper exists; an override file changes the instance's
    /// <see cref="Options"/>, and the figure shown is refreshed from there once one is built.
    /// </summary>
    public static long ApproximateDownloadBytes => SpeechModelOptions.Default.ApproximateDownloadBytes;

    /// <summary>
    /// Shared by every bootstrapper that is not handed a client of its own. The settings window,
    /// the warm start, and the harness each construct one of these, and a per-instance client
    /// would leak a connection pool every time; a download this size also wants a long timeout,
    /// which is the only reason the default client is not used.
    /// </summary>
    private static readonly HttpClient SharedHttpClient = new() { Timeout = TimeSpan.FromMinutes(10) };

    private readonly HttpClient _httpClient;
    private readonly string _modelsDirectory;

    public SpeechModelBootstrapper(
        HttpClient? httpClient = null,
        string? modelsDirectory = null,
        SpeechModelOptions? options = null)
    {
        _httpClient = httpClient ?? SharedHttpClient;
        _modelsDirectory = modelsDirectory ?? AppPaths.ModelsDirectory;
        Options = options ?? SpeechModelOptions.Load();
    }

    /// <summary>Where the files come from and how small is too small. See <see cref="SpeechModelOptions"/>.</summary>
    public SpeechModelOptions Options { get; }

    public string ModelId => Options.ModelId;

    public string EncoderPath => Path.Combine(_modelsDirectory, Options.ModelId + "-encoder.int8.onnx");

    public string DecoderPath => Path.Combine(_modelsDirectory, Options.ModelId + "-decoder.int8.onnx");

    public string JoinerPath => Path.Combine(_modelsDirectory, Options.ModelId + "-joiner.int8.onnx");

    public string TokensPath => Path.Combine(_modelsDirectory, Options.ModelId + "-tokens.txt");

    public string VadPath => Path.Combine(_modelsDirectory, "silero_vad.onnx");

    public string ModelsDirectory => _modelsDirectory;

    public IReadOnlyList<string> ModelFilePaths =>
    [
        EncoderPath,
        DecoderPath,
        JoinerPath,
        TokensPath,
        VadPath,
    ];

    public long InstalledBytes => ModelFilePaths
        .Where(File.Exists)
        .Sum(path => new FileInfo(path).Length);

    public bool HasModelFiles => ModelFilePaths
        .SelectMany(path => new[] { path, path + ".tmp" })
        .Any(File.Exists);

    /// <summary>
    /// Whether every file is already on disk, so dictation can start without a download. Read at
    /// startup to decide whether loading the engine needs the user's consent first.
    /// </summary>
    public bool IsDownloaded =>
        IsUsableFile(EncoderPath, Options.MinimumEncoderBytes)
        && IsUsableFile(DecoderPath, Options.MinimumDecoderBytes)
        && IsUsableFile(JoinerPath, Options.MinimumJoinerBytes)
        && IsUsableFile(TokensPath, Options.MinimumTokensBytes)
        && IsUsableFile(VadPath, Options.MinimumVadBytes);

    public void DeleteModelFiles()
    {
        var root = Path.GetFullPath(_modelsDirectory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;

        foreach (var path in ModelFilePaths.SelectMany(path => new[] { path, path + ".tmp" }))
        {
            var fullPath = Path.GetFullPath(path);
            if (!fullPath.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Speech model files must be inside the models directory.");

            if (File.Exists(fullPath))
                File.Delete(fullPath);
        }
    }

    public async Task<SpeechModelFiles> EnsureAsync(
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        AppPaths.EnsureCreated();
        Directory.CreateDirectory(_modelsDirectory);

        using var activity = SpeechDiagnostics.StartActivity(SpeechDiagnostics.ModelEnsureActivity);
        activity?.SetTag("speech.model", Options.ModelId);
        activity?.SetTag("speech.model.already_present", IsDownloaded);

        // Weighted by size so the bar moves at roughly a constant rate: the encoder is the
        // download, and the other three together are rounding error.
        await DownloadIfNeededAsync(Options.EncoderUrl, EncoderPath, Options.MinimumEncoderBytes, null, ScaleProgress(progress, 0.0, 0.92), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(Options.DecoderUrl, DecoderPath, Options.MinimumDecoderBytes, null, ScaleProgress(progress, 0.92, 0.94), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(Options.JoinerUrl, JoinerPath, Options.MinimumJoinerBytes, null, ScaleProgress(progress, 0.94, 0.95), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(Options.TokensUrl, TokensPath, Options.MinimumTokensBytes, null, ScaleProgress(progress, 0.95, 0.96), cancellationToken).ConfigureAwait(false);
        await DownloadIfNeededAsync(Options.VadUrl, VadPath, Options.MinimumVadBytes, Options.VadSha256, ScaleProgress(progress, 0.96, 1.0), cancellationToken).ConfigureAwait(false);
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

        MoveIntoPlace(tempPath, path, activity);
        progress?.Report(1.0);
    }

    /// <summary>
    /// Publishes a completed download under its final name, retrying briefly if the file is still
    /// locked.
    ///
    /// <para>
    /// Defender's real-time protection scans a file when the last handle on it closes, and an
    /// on-access scan of a 60 MB model holds the file open for long enough that the move that
    /// follows immediately can lose the race and fail with <see cref="IOException"/> or
    /// <see cref="UnauthorizedAccessException"/>. The lock is transient, so a short backoff clears
    /// it; a lock that outlasts the backoff is a real failure (a quarantined file, a locked
    /// directory) and is reported rather than retried forever.
    /// </para>
    /// <para>
    /// No unblocking is needed, and none is attempted. A Mark-of-the-Web -
    /// the <c>Zone.Identifier</c> stream that <c>Unblock-File</c> removes - is written by the
    /// Attachment Execution Service on behalf of browsers and mail clients, not by the file system,
    /// so a file written here by <see cref="HttpClient"/> never carries one. Writing one of our own
    /// would be adding the restriction, and clearing a mark we did not set would be the wrong thing
    /// for this code to be doing.
    /// </para>
    /// </summary>
    private static void MoveIntoPlace(string tempPath, string path, Activity? activity)
    {
        const int attempts = 5;
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                File.Move(tempPath, path, overwrite: true);
                if (attempt > 1)
                    activity?.SetTag("file.move_attempts", attempt);
                return;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException && attempt < attempts)
            {
                Thread.Sleep(100 * attempt);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                activity?.SetStatus(ActivityStatusCode.Error, ex.GetType().Name);
                TryDelete(tempPath);
                throw new SpeechModelDownloadException(
                    $"The speech model was downloaded but could not be moved to {path}. "
                    + "Another process - commonly antivirus real-time scanning - is holding the file open.",
                    ex);
            }
        }
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
