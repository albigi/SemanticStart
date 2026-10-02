using System.Reflection;
using System.Runtime.CompilerServices;
using System.Windows.Threading;
using Microsoft.Extensions.Logging.Abstractions;
using SemanticStart.App;
using SemanticStart.Core.Speech;

namespace SemanticStart.Tests;

public sealed class DictationCancellationTests
{
    [Fact]
    public void CancellingPreparationNeverOpensTheMicrophoneAfterTheModelIsReady() => OnDispatcher(async () =>
    {
        var settings = new AppSettings { DictationEnabled = true, SearchDebounceMilliseconds = 60_000 };
        var viewModel = new OverlayViewModel(new SemanticSearchService(settings), new IconProvider(), settings);
        var ready = new TaskCompletionSource<DictationEngine?>();
        var microphoneCreations = 0;
        using var engine = new DictationEngine(new LateTranscriber(), new TestVad(), () =>
        {
            microphoneCreations++;
            return new UnusedMicrophone();
        });
        using var controller = CreateController(viewModel, settings);
        SetField(controller, "_warmStart", ready.Task);

        var start = StartAsync(controller);
        Assert.True(viewModel.IsPreparingDictation);
        controller.Cancel();
        viewModel.Clear();
        ready.SetResult(engine);
        await start;

        Assert.Equal(0, microphoneCreations);
        Assert.False(controller.IsListening);
        Assert.False(viewModel.IsPreparingDictation);
        Assert.Equal(OverlayViewModel.IdleStatus, viewModel.Status);
    });

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void CancelDiscardsQueuedAndLateTranscriptsButStopStillFinalizes(bool abandon) => OnDispatcher(async () =>
    {
        var settings = new AppSettings { DictationEnabled = true, SearchDebounceMilliseconds = 60_000 };
        var viewModel = new OverlayViewModel(new SemanticSearchService(settings), new IconProvider(), settings);
        using var engine = new DictationEngine(new LateTranscriber(), new TestVad(), () => new UnusedMicrophone());
        using var controller = CreateController(viewModel, settings);
        SetField(controller, "_engine", engine);

        var start = StartAsync(controller);
        Assert.True(controller.IsListening);
        if (abandon)
        {
            controller.Cancel();
            viewModel.Clear();
            viewModel.Query = "keyboard query";
        }
        else
        {
            controller.Stop();
        }

        await start;
        await Dispatcher.CurrentDispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);

        Assert.Equal(abandon ? "keyboard query" : "late final", viewModel.Query);
        Assert.False(viewModel.IsListening);
        viewModel.Clear();
    });

    private static DictationController CreateController(OverlayViewModel viewModel, AppSettings settings) =>
        new(Dispatcher.CurrentDispatcher, viewModel, settings, () => false, () => { },
            NullLogger<DictationController>.Instance);

    private static Task StartAsync(DictationController controller) =>
        (Task)typeof(DictationController).GetMethod("StartAsync", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(controller, null)!;

    private static void SetField(DictationController controller, string name, object value) =>
        typeof(DictationController).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(controller, value);

    private static void OnDispatcher(Func<Task> test)
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            var dispatcher = Dispatcher.CurrentDispatcher;
            SynchronizationContext.SetSynchronizationContext(new DispatcherSynchronizationContext(dispatcher));
            dispatcher.InvokeAsync(async () =>
            {
                try
                {
                    await test();
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
                finally
                {
                    dispatcher.BeginInvokeShutdown(DispatcherPriority.Background);
                }
            });
            Dispatcher.Run();
        }) { IsBackground = true };
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(10)), "The dictation test timed out.");
        Assert.Null(failure);
    }

    private sealed class LateTranscriber : ISpeechTranscriber
    {
        public SpeechProviderMetadata Metadata { get; } = new("test", "Test", "test", "en-US", true, false, 0);

        public async IAsyncEnumerable<SpeechTranscript> TranscribeAsync(
            IAsyncEnumerable<ReadOnlyMemory<float>> audio,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var stopped = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            using var registration = cancellationToken.Register(() => stopped.TrySetResult());
            yield return new PartialTranscript("queued partial");
            await stopped.Task;
            yield return new FinalTranscript("late final");
        }

        public void Dispose() { }
    }

    private sealed class TestVad : IVoiceActivityDetector
    {
        public int FrameSamples => 512;
        public TimeSpan FrameDuration => TimeSpan.FromMilliseconds(32);
        public float Process(ReadOnlySpan<float> frame) => 0;
        public void Reset() { }
        public void Dispose() { }
    }

    private sealed class UnusedMicrophone : IAudioCaptureSource
    {
        public int SampleRate => 16_000;
        public IAsyncEnumerable<ReadOnlyMemory<float>> CaptureAsync(CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("The scripted transcriber must not open a microphone.");
        public void Dispose() { }
    }
}
