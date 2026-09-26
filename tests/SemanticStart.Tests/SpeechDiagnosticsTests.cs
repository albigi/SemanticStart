using System.Diagnostics;
using SemanticStart.Core.Speech;

namespace SemanticStart.Tests;

/// <summary>
/// Dictation is the one feature in the app whose failures are mostly invisible: a microphone that
/// will not stop, a temp file that will not delete, a provider that would not load. None of those
/// reach a caller, so the rule is that everything handled internally is announced instead of
/// discarded. These tests hold that rule and the trace source the app listens on.
/// </summary>
public sealed class SpeechDiagnosticsTests
{
    [Fact]
    public void ReportFailureCarriesTheExceptionRatherThanItsMessage()
    {
        var reports = new List<SpeechDiagnosticEvent>();
        void Handler(SpeechDiagnosticEvent report) => reports.Add(report);

        var thrown = new InvalidOperationException("the device went away");

        SpeechDiagnostics.Reported += Handler;
        try
        {
            SpeechDiagnostics.ReportFailure("capture.stop", "Stopping the microphone failed.", thrown);
        }
        finally
        {
            SpeechDiagnostics.Reported -= Handler;
        }

        var failure = Assert.Single(reports, report => report.Operation == "capture.stop");
        Assert.True(failure.IsError);

        // The whole exception, so the log keeps a stack: a swallowed native error reduced to its
        // message is a failure nobody can act on.
        Assert.Same(thrown, failure.Exception);
    }

    [Fact]
    public void AThrowingSubscriberDoesNotBreakThePathBeingReportedOn()
    {
        void Throws(SpeechDiagnosticEvent _) => throw new InvalidOperationException("bad handler");

        SpeechDiagnostics.Reported += Throws;
        try
        {
            SpeechDiagnostics.Report("model.download", "Downloading.");
            SpeechDiagnostics.ReportFailure("model.cleanup", "Could not delete.", new IOException("locked"));
        }
        finally
        {
            SpeechDiagnostics.Reported -= Throws;
        }
    }

    [Fact]
    public void StartActivityIsFreeWhenNothingIsListening()
    {
        // No listener, no span: the instrumentation on the dictation path has to cost nothing in a
        // build where nobody has attached to it.
        using var activity = SpeechDiagnostics.StartActivity(SpeechDiagnostics.ListenActivity);

        Assert.Null(activity);
    }

    [Fact]
    public void ListenersSeeSpansWithTheirTags()
    {
        var stopped = new List<Activity>();
        using var listener = new ActivityListener
        {
            ShouldListenTo = source => source.Name == SpeechDiagnostics.ActivitySourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = stopped.Add,
        };

        ActivitySource.AddActivityListener(listener);

        using (var activity = SpeechDiagnostics.StartActivity(SpeechDiagnostics.EngineStartActivity))
        {
            Assert.NotNull(activity);
            activity.SetTag("speech.provider", "sherpa-onnx-streaming-zipformer");
        }

        var span = Assert.Single(stopped, a => a.OperationName == SpeechDiagnostics.EngineStartActivity);
        Assert.Equal("sherpa-onnx-streaming-zipformer", span.GetTagItem("speech.provider"));
    }

    [Fact]
    public async Task AProviderThatFailsToLoadIsReportedAndNotOnlyRejected()
    {
        var thrown = new InvalidOperationException("onnxruntime.dll could not be loaded");
        var broken = new TestSpeechTranscriberProvider(
            "broken-for-diagnostics",
            create: () => throw thrown);
        var selector = new SpeechEngineSelector([broken]);

        var reports = new List<SpeechDiagnosticEvent>();
        void Handler(SpeechDiagnosticEvent report) => reports.Add(report);

        SpeechDiagnostics.Reported += Handler;
        SpeechEngineStartupResult startup;
        try
        {
            startup = await selector.StartAsync();
        }
        finally
        {
            SpeechDiagnostics.Reported -= Handler;
        }

        Assert.False(startup.IsAvailable);
        Assert.Same(thrown, Assert.Single(startup.Rejected).Exception);

        // Carrying on to the next provider is correct; doing it silently is not.
        Assert.Contains(reports, report => report.IsError && ReferenceEquals(report.Exception, thrown));
    }
}
