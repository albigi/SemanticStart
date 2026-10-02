using SemanticStart.App;

namespace SemanticStart.Tests;

public sealed class DictationInputTests : IDisposable
{
    private readonly OverlayViewModel _viewModel;

    public DictationInputTests()
    {
        var settings = new AppSettings { SearchDebounceMilliseconds = 60_000 };
        _viewModel = new OverlayViewModel(new SemanticSearchService(settings), new IconProvider(), settings);
    }

    [Theory]
    [InlineData("", "notepad")]
    [InlineData("open", "open notepad")]
    public async Task RetryReplacesUneditedFinalAndKeepsTypedPrefix(string prefix, string expected)
    {
        _viewModel.Query = prefix;
        _viewModel.BeginDictation();
        await _viewModel.ApplyFinalTranscriptAsync("word pad");
        _viewModel.EndDictation();

        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("note");
        await _viewModel.ApplyFinalTranscriptAsync("notepad");

        Assert.Equal(expected, _viewModel.Query);
    }

    [Fact]
    public void RetryAlsoReplacesAPartialFromAnInterruptedTurn()
    {
        _viewModel.Query = "open";
        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("wrong");
        _viewModel.EndDictation();
        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("notepad");

        Assert.Equal("open notepad", _viewModel.Query);
    }

    [Theory]
    [InlineData("open corrected")]
    [InlineData("")]
    public async Task KeyboardEditIgnoresRemainingTranscriptsAndBecomesTheNextPrefix(string edited)
    {
        _viewModel.Query = "open";
        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("wrong");
        _viewModel.Query = edited;
        _viewModel.ApplyPartialTranscript("late partial");
        await _viewModel.ApplyFinalTranscriptAsync("late final");

        Assert.Equal(edited, _viewModel.Query);

        _viewModel.EndDictation();
        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("notepad");
        Assert.Equal(edited.Length == 0 ? "notepad" : edited + " notepad", _viewModel.Query);
    }

    [Fact]
    public void AnEditBeforeTheFirstPartialAlsoOwnsTheQuery()
    {
        _viewModel.BeginDictation();
        _viewModel.Query = "typed while listening";
        _viewModel.ApplyPartialTranscript("ignored");

        Assert.Equal("typed while listening", _viewModel.Query);
    }

    [Fact]
    public async Task EditingAfterDictationPreservesTheWholeEditedQueryOnRetry()
    {
        _viewModel.BeginDictation();
        await _viewModel.ApplyFinalTranscriptAsync("notepad");
        _viewModel.EndDictation();
        _viewModel.Query = "notepad corrected";
        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("settings");

        Assert.Equal("notepad corrected settings", _viewModel.Query);
    }

    [Fact]
    public async Task ClearResetsListeningPreparationAndOwnershipAndIgnoresLateTranscripts()
    {
        _viewModel.Query = "open";
        _viewModel.ReportDictationPreparing();
        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("wrong");
        _viewModel.Clear();
        _viewModel.ApplyPartialTranscript("late partial");
        await _viewModel.ApplyFinalTranscriptAsync("late final");

        Assert.Empty(_viewModel.Query);
        Assert.False(_viewModel.IsListening);
        Assert.False(_viewModel.IsPreparingDictation);
        Assert.Equal(0, _viewModel.MicrophoneLevel);

        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("notepad");
        Assert.Equal("notepad", _viewModel.Query);
    }

    [Fact]
    public async Task AnEmptyRetryDoesNotEraseThePreviousQuery()
    {
        _viewModel.BeginDictation();
        await _viewModel.ApplyFinalTranscriptAsync("notepad");
        _viewModel.EndDictation();
        _viewModel.BeginDictation();
        _viewModel.ApplyPartialTranscript("");
        await _viewModel.ApplyFinalTranscriptAsync(" ");

        Assert.Equal("notepad", _viewModel.Query);
    }

    public void Dispose() => _viewModel.Clear();
}
