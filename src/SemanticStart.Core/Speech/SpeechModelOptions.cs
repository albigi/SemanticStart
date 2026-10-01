using System.Text.Json;

namespace SemanticStart.Core.Speech;

/// <summary>
/// Where the speech models come from and how small a file is too small to be one of them.
///
/// <para>
/// Separated from <see cref="SpeechModelBootstrapper"/> so the download logic holds no literals:
/// a different mirror, an air-gapped internal host, or a newer Zipformer revision is then a data
/// change rather than a code change. The defaults are the published upstream locations, so the
/// ordinary case needs no file at all.
/// </para>
/// <para>
/// Overrides are read from <c>speech-model.json</c> in the app's data directory with
/// <see cref="JsonSerializer"/>, which is the same mechanism <c>settings.json</c> already uses.
/// <c>Microsoft.Extensions.Configuration</c> and <c>IOptions&lt;T&gt;</c> are deliberately not used
/// here: both are resolved from a host's service provider, and this app has no container - it is
/// composed by hand in <c>App.OnStartup</c> - so they would add a dependency without adding the
/// lifetime management that is the reason to take one.
/// </para>
/// </summary>
public sealed record SpeechModelOptions
{
    /// <summary>The file read by <see cref="Load"/> when it exists.</summary>
    public const string FileName = "speech-model.json";

    private const string ZipformerBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/";

    /// <summary>
    /// Identifies the model on disk and in traces. Part of every downloaded file's name, so two
    /// revisions can sit side by side and switching back does not re-download.
    /// </summary>
    public string ModelId { get; init; } = "sherpa-onnx-streaming-zipformer-en-2023-06-26";

    public string EncoderUrl { get; init; } = ZipformerBaseUrl + "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";

    public string DecoderUrl { get; init; } = ZipformerBaseUrl + "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";

    public string JoinerUrl { get; init; } = ZipformerBaseUrl + "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx";

    public string TokensUrl { get; init; } = ZipformerBaseUrl + "tokens.txt";

    /// <summary>Silero VAD v5.1, pinned to a tag so a released build's model cannot change.</summary>
    public string VadUrl { get; init; } =
        "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/src/silero_vad/data/silero_vad.onnx";

    /// <summary>
    /// SHA-256 of silero_vad.onnx at tag v5.1. Verified because the file is small enough that
    /// hashing costs nothing and it is fetched from a raw file URL rather than a release asset.
    /// Null disables the check, which is what pointing <see cref="VadUrl"/> at another build means.
    /// </summary>
    public string? VadSha256 { get; init; } = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f";

    /// <summary>
    /// Size floors, in bytes. A truncated download is the failure these catch: it leaves a file
    /// that exists, so existence alone would be taken as "already downloaded" forever. The
    /// Zipformer files are checked by size rather than by hash because upstream publishes no
    /// hashes, and one recorded here would be a claim about one download rather than the model.
    /// </summary>
    public long MinimumEncoderBytes { get; init; } = 50_000_000;

    public long MinimumDecoderBytes { get; init; } = 500_000;

    public long MinimumJoinerBytes { get; init; } = 100_000;

    public long MinimumTokensBytes { get; init; } = 1_000;

    public long MinimumVadBytes { get; init; } = 1_000_000;

    /// <summary>What the user is told they are about to download, before it starts.</summary>
    public long ApproximateDownloadBytes { get; init; } = 75L * 1024 * 1024;

    /// <summary>The upstream defaults, used whenever no override file is present.</summary>
    public static SpeechModelOptions Default { get; } = new();

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    /// <summary>
    /// Reads <see cref="FileName"/> from <paramref name="directory"/>, falling back to
    /// <see cref="Default"/> when it is absent. Absent is the normal case, so it is not an error;
    /// a file that is present and unreadable is, because silently downloading from somewhere other
    /// than the operator asked for is worse than refusing to start.
    /// </summary>
    public static SpeechModelOptions Load(string? directory = null)
    {
        var path = Path.Combine(directory ?? AppPaths.Root, FileName);
        if (!File.Exists(path))
            return Default;

        try
        {
            return JsonSerializer.Deserialize<SpeechModelOptions>(File.ReadAllText(path), JsonOptions) ?? Default;
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            throw new SpeechModelDownloadException($"Could not read the speech model configuration at {path}.", ex);
        }
    }
}
