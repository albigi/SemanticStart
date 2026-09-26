using SemanticStart.Core.Speech;

namespace SemanticStart.Tests;

/// <summary>
/// Only the part of the bootstrapper that can be answered from disk: where the files go, and
/// whether they are all there. That question is asked at startup to decide whether dictation can
/// be loaded without first asking the user to authorise a 75 MB download, so a wrong answer either
/// downloads without consent or reports a model missing that is sitting right there.
///
/// <para>
/// Nothing here reaches the network. The bootstrapper takes its models directory, so the files are
/// faked in a scratch directory at exactly the sizes the size floor is checking for - the floor
/// exists to reject a truncated download, and a zero-byte placeholder has to fail it.
/// </para>
/// </summary>
public sealed class SpeechModelBootstrapperTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ss-speech-" + Guid.NewGuid().ToString("N"));

    public SpeechModelBootstrapperTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void NothingOnDiskMeansNothingIsDownloaded()
    {
        Assert.False(new SpeechModelBootstrapper(modelsDirectory: _dir).IsDownloaded);
    }

    [Fact]
    public void EveryFilePresentAndFullSizeMeansDownloaded()
    {
        var bootstrapper = new SpeechModelBootstrapper(modelsDirectory: _dir);
        WriteFullSizeModel(bootstrapper);

        Assert.True(bootstrapper.IsDownloaded);
    }

    /// <summary>
    /// One missing file is a half-downloaded model, which loads as an error deep inside the ONNX
    /// runtime rather than as "the model is not here".
    /// </summary>
    [Theory]
    [InlineData("encoder")]
    [InlineData("decoder")]
    [InlineData("joiner")]
    [InlineData("tokens")]
    [InlineData("vad")]
    public void OneMissingFileMeansNotDownloaded(string missing)
    {
        var bootstrapper = new SpeechModelBootstrapper(modelsDirectory: _dir);
        WriteFullSizeModel(bootstrapper);
        File.Delete(PathFor(bootstrapper, missing));

        Assert.False(bootstrapper.IsDownloaded);
    }

    /// <summary>
    /// The failure this guards is an interrupted download left in place: a file that exists, is
    /// far too small, and would otherwise be taken for a complete model forever.
    /// </summary>
    [Theory]
    [InlineData("encoder")]
    [InlineData("vad")]
    public void ATruncatedFileMeansNotDownloaded(string truncated)
    {
        var bootstrapper = new SpeechModelBootstrapper(modelsDirectory: _dir);
        WriteFullSizeModel(bootstrapper);
        WriteFile(PathFor(bootstrapper, truncated), 1_024);

        Assert.False(bootstrapper.IsDownloaded);
    }

    /// <summary>
    /// The files live in the directory the caller named, under names carrying the model id, so two
    /// models can share the one models directory the embedding model already uses.
    /// </summary>
    [Fact]
    public void ModelFilesArePlacedInTheGivenDirectory()
    {
        var bootstrapper = new SpeechModelBootstrapper(modelsDirectory: _dir);

        foreach (var path in AllPaths(bootstrapper))
            Assert.Equal(_dir, Path.GetDirectoryName(path));

        Assert.StartsWith(SpeechModelBootstrapper.ModelId, Path.GetFileName(bootstrapper.EncoderPath), StringComparison.Ordinal);
        Assert.Equal("silero_vad.onnx", Path.GetFileName(bootstrapper.VadPath));

        // Distinct names, or one download overwrites the last.
        Assert.Equal(5, AllPaths(bootstrapper).Distinct(StringComparer.OrdinalIgnoreCase).Count());
    }

    /// <summary>
    /// The VAD model is the one file fetched from a raw file URL rather than release hosting, and
    /// the checksum is what makes that safe, so the comparison itself is worth pinning.
    /// </summary>
    [Fact]
    public void TheChecksumComparisonAcceptsOnlyTheExpectedContent()
    {
        var path = Path.Combine(_dir, "checksum.bin");
        File.WriteAllText(path, "semanticstart");

        // Not the hash of anything: the point is that a file whose contents do not match is
        // rejected rather than accepted because it was the right length.
        const string wrong = "0000000000000000000000000000000000000000000000000000000000000000";

        Assert.False(SpeechModelBootstrapper.MatchesSha256(path, wrong));

        // Spelled in lower case as well, because the published hash is lower case while
        // Convert.ToHexString produces upper.
        Assert.True(SpeechModelBootstrapper.MatchesSha256(path, Sha256Of(path)));
        Assert.True(SpeechModelBootstrapper.MatchesSha256(path, Sha256Of(path).ToLowerInvariant()));
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    private static string Sha256Of(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));
    }

    private static string[] AllPaths(SpeechModelBootstrapper bootstrapper) =>
    [
        bootstrapper.EncoderPath,
        bootstrapper.DecoderPath,
        bootstrapper.JoinerPath,
        bootstrapper.TokensPath,
        bootstrapper.VadPath,
    ];

    private static string PathFor(SpeechModelBootstrapper bootstrapper, string file) => file switch
    {
        "encoder" => bootstrapper.EncoderPath,
        "decoder" => bootstrapper.DecoderPath,
        "joiner" => bootstrapper.JoinerPath,
        "tokens" => bootstrapper.TokensPath,
        "vad" => bootstrapper.VadPath,
        _ => throw new ArgumentOutOfRangeException(nameof(file), file, "Unknown model file."),
    };

    /// <summary>
    /// Files at the size the bootstrapper's floors demand. Written by length rather than by
    /// content: the encoder floor is 50 MB, and setting the length leaves the filesystem to
    /// account for the space instead of this test writing it a megabyte at a time.
    /// </summary>
    private static void WriteFullSizeModel(SpeechModelBootstrapper bootstrapper)
    {
        WriteFile(bootstrapper.EncoderPath, 50_000_000);
        WriteFile(bootstrapper.DecoderPath, 500_000);
        WriteFile(bootstrapper.JoinerPath, 100_000);
        WriteFile(bootstrapper.TokensPath, 1_000);
        WriteFile(bootstrapper.VadPath, 1_000_000);
    }

    private static void WriteFile(string path, long length)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        stream.SetLength(length);
    }
}
