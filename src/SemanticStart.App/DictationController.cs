using System.Diagnostics;
using System.Windows.Threading;
using SemanticStart.Core;
using SemanticStart.Core.Speech;

namespace SemanticStart.App;

/// <summary>
/// Connects the dictation hotkey to the overlay: keeps one warm recogniser for the life of the
/// process, runs a listening turn on demand, and puts transcripts into the query box.
///
/// <para>
/// The engine is created once at startup rather than on the first press. Loading the model is
/// hundreds of milliseconds of work that would otherwise land in front of a user who has already
/// started talking, and a first utterance that loses its opening words is worse than one that never
/// started. Everything on the hotkey path here is bookkeeping.
/// </para>
///
/// <para>
/// Speech never leaves the machine: the transcriber is an ONNX model reading the local microphone.
/// See <see cref="ISpeechTranscriber"/> for why the platform's own recognisers are not used.
/// </para>
/// </summary>
public sealed class DictationController : IDisposable
{
    /// <summary>
    /// How often the dictation chord is checked for release in push-to-talk mode. Short enough to
    /// feel immediate and long enough to be free; release is noticed within one interval.
    /// </summary>
    private static readonly TimeSpan PushToTalkPollInterval = TimeSpan.FromMilliseconds(40);

    private readonly Dispatcher _dispatcher;
    private readonly OverlayViewModel _viewModel;
    private readonly Func<bool> _isHotKeyHeld;
    private readonly Action _showOverlay;
    private readonly SemaphoreSlim _startGate = new(1, 1);

    /// <summary>
    /// Cancels the warm start. Loading the engine can involve a model download, so without this a
    /// shutdown during first run would leave an HTTP read running against a temp file until
    /// HttpClient's own timeout, and would finish by constructing a recognizer nobody wants.
    /// </summary>
    private readonly CancellationTokenSource _startup = new();

    private AppSettings _settings;
    private DictationEngine? _engine;
    private Task<DictationEngine?>? _warmStart;
    private CancellationTokenSource? _session;
    private DispatcherTimer? _pushToTalkTimer;
    private string? _unavailableReason;
    private bool _disposed;

    public DictationController(
        Dispatcher dispatcher,
        OverlayViewModel viewModel,
        AppSettings settings,
        Func<bool> isHotKeyHeld,
        Action showOverlay)
    {
        _dispatcher = dispatcher;
        _viewModel = viewModel;
        _settings = settings;
        _isHotKeyHeld = isHotKeyHeld;
        _showOverlay = showOverlay;
    }

    /// <summary>Whether a listening turn is in progress.</summary>
    public bool IsListening => _session is not null;

    public void ApplySettings(AppSettings settings)
    {
        var wasEnabled = _settings.DictationEnabled;
        _settings = settings;

        if (settings.DictationEnabled && !wasEnabled)
            _ = WarmStartAsync();
        else if (!settings.DictationEnabled)
            Stop();
    }

    /// <summary>
    /// Loads the recogniser, once, in the background. Failure is remembered rather than thrown:
    /// the rest of the app works without dictation, and the reason is what the user needs when
    /// they press the hotkey and nothing happens.
    /// </summary>
    public Task WarmStartAsync()
    {
        if (_disposed || !_settings.DictationEnabled)
            return Task.CompletedTask;

        // Task.Run rather than TaskFactory.StartNew: this is called on the UI thread, and
        // StartNew queues to TaskScheduler.Current - which is the UI scheduler whenever the caller
        // is itself inside a dispatcher-marshalled continuation - so the model load would run on
        // the STA thread it exists to stay off. Task.Run always means TaskScheduler.Default, and
        // unwraps the async delegate instead of handing back a Task<Task>. LongRunning is wrong
        // here for the same reason: the load is mostly awaited I/O, and the CPU-bound stretch is
        // short enough not to be worth a dedicated thread.
        _warmStart ??= Task.Run(CreateEngineAsync);
        return _warmStart;
    }

    /// <summary>
    /// The dictation hotkey. Opens the overlay and starts listening; in toggle mode a second press
    /// stops. Push-to-talk stops on release instead, polled rather than hooked - see
    /// <see cref="ActivationManager.IsDictationHotKeyHeld"/>.
    /// </summary>
    public void Toggle()
    {
        if (_disposed)
            return;

        _showOverlay();

        if (IsListening)
        {
            if (!_settings.DictationPushToTalk)
                Stop();
            return;
        }

        _ = StartAsync();
    }

    public void Stop()
    {
        StopPushToTalkPolling();

        // Cancelled but not disposed: the listening task is still holding this token, and
        // disposing it underneath ONNX and the capture wrapper is how a cancellation turns into an
        // ObjectDisposedException. RunSessionAsync's finally owns the disposal.
        _session?.Cancel();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        Stop();
        _startup.Cancel();

        // The start gate is deliberately not disposed: a start that is still between its wait and
        // its release would then throw on a background task during shutdown, which is a crash log
        // for no benefit. Letting it be collected costs nothing.

        // The warm-start task owns the engine until it completes, so disposing here would race it.
        // Hand the disposal to whichever finishes last.
        if (_warmStart is { } warmStart)
        {
            _ = warmStart.ContinueWith(
                task =>
                {
                    // Nothing awaits this continuation, so an exception here would be an
                    // unobserved task exception raised long after shutdown, attributed to nothing.
                    // Releasing native ONNX sessions is exactly where that would happen.
                    try
                    {
                        task.Result?.Dispose();
                    }
                    catch (Exception ex)
                    {
                        Log.Error(ex, "Disposing the dictation engine failed");
                    }
                },
                // The load itself is cancelled above, by _startup. This token is the
                // continuation's own, and must stay None: a cancelled continuation would skip the
                // disposal and leak the native ONNX sessions it exists to release.
                CancellationToken.None,
                TaskContinuationOptions.OnlyOnRanToCompletion,
                TaskScheduler.Default).ContinueWith(_ => _startup.Dispose(), TaskScheduler.Default);
        }
        else
        {
            try
            {
                _engine?.Dispose();
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Disposing the dictation engine failed");
            }

            _startup.Dispose();
        }
    }

    private async Task<DictationEngine?> CreateEngineAsync()
    {
        var started = Stopwatch.StartNew();
        try
        {
            var models = new SpeechModelBootstrapper();
            var selector = new SpeechEngineSelector([new SherpaOnnxSpeechProvider(models)]);
            var startup = await selector.StartAsync(cancellationToken: _startup.Token).ConfigureAwait(false);

            // Every provider that said no, with its reason, rather than only the first one that
            // ends up in the status line: with a second engine this is the only record of why the
            // one that was expected to win did not.
            foreach (var rejection in startup.Rejected)
            {
                if (rejection.Exception is { } failure)
                    Log.Error(failure, $"Speech provider '{rejection.Metadata.Id}' was rejected: {rejection.Reason}");
                else
                    Log.Info($"Speech provider '{rejection.Metadata.Id}' was rejected: {rejection.Reason}");
            }

            if (!startup.IsAvailable || startup.Transcriber is null)
            {
                _unavailableReason = startup.FailureReason;
                Log.Info($"Dictation is unavailable: {_unavailableReason}");
                return null;
            }

            var engine = new DictationEngine(
                startup.Transcriber,
                new SileroVoiceActivityDetector(models.VadPath),
                () => new WasapiMicrophoneCapture());

            _engine = engine;
            Log.Info($"Dictation ready: {engine.Metadata.DisplayName} ({engine.Metadata.ModelId}) in {started.ElapsedMilliseconds} ms.");
            return engine;
        }
        catch (OperationCanceledException) when (_startup.IsCancellationRequested)
        {
            // Shutdown, not a failure. Recorded rather than dropped so a log that ends here is
            // distinguishable from one where the load simply never finished.
            Log.Info($"The dictation engine load was abandoned at shutdown after {started.ElapsedMilliseconds} ms.");
            return null;
        }
        catch (Exception ex)
        {
            _unavailableReason = "The speech model could not be loaded; see the log for details.";
            Log.Error(ex, $"Failed to start the dictation engine after {started.ElapsedMilliseconds} ms");
            return null;
        }
    }

    private async Task StartAsync()
    {
        if (!await _startGate.WaitAsync(0).ConfigureAwait(true))
            return;

        CancellationTokenSource? session = null;
        try
        {
            var engine = await WarmStartEngineAsync().ConfigureAwait(true);
            if (engine is null)
            {
                _viewModel.EndDictation(_unavailableReason ?? "Dictation is unavailable on this machine.");
                return;
            }

            if (_disposed || IsListening)
                return;

            session = new CancellationTokenSource();
            _session = session;
            _viewModel.BeginDictation();
            StartPushToTalkPolling();
        }
        finally
        {
            _startGate.Release();
        }

        if (session is null)
            return;

        await RunSessionAsync(session).ConfigureAwait(true);
    }

    private async Task<DictationEngine?> WarmStartEngineAsync()
    {
        if (_engine is { } ready)
            return ready;

        _viewModel.ReportStatus("Preparing dictation\u2026");
        _warmStart ??= Task.Run(CreateEngineAsync);
        return await _warmStart.ConfigureAwait(true);
    }

    private async Task RunSessionAsync(CancellationTokenSource session)
    {
        var options = new DictationOptions
        {
            TrailingSilence = TimeSpan.FromMilliseconds(_settings.DictationTrailingSilenceMilliseconds),
        };

        try
        {
            await _engine!.ListenAsync(
                    options,
                    transcript => _dispatcher.InvokeAsync(() => OnTranscript(transcript)),
                    level => _dispatcher.InvokeAsync(() => _viewModel.ReportMicrophoneLevel(level)),
                    session.Token)
                .ConfigureAwait(true);

            _viewModel.EndDictation();
        }
        catch (OperationCanceledException)
        {
            // Expected - it is how Stop, push-to-talk release and hiding the overlay all end a
            // session - but a session that ends with no trace at all is indistinguishable from one
            // that never started, so it is still recorded.
            Log.Trace("Dictation session was cancelled.");
            _viewModel.EndDictation();
        }
        catch (MicrophoneUnavailableException ex)
        {
            // Never fail silently here: a microphone that cannot be opened looks identical to a
            // recogniser that heard nothing, and only one of the two is fixable by the user.
            Log.Error(ex, "The microphone could not be opened for dictation");
            _viewModel.EndDictation(ex.Message);
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Dictation failed");
            _viewModel.EndDictation("Dictation failed; see the log for details.");
        }
        finally
        {
            StopPushToTalkPolling();
            if (ReferenceEquals(_session, session))
            {
                _session = null;
                session.Dispose();
            }
        }
    }

    private void OnTranscript(SpeechTranscript transcript)
    {
        if (transcript is FinalTranscript)
            _ = _viewModel.ApplyFinalTranscriptAsync(transcript.Text);
        else
            _viewModel.ApplyPartialTranscript(transcript.Text);
    }

    private void StartPushToTalkPolling()
    {
        if (!_settings.DictationPushToTalk)
            return;

        _pushToTalkTimer = new DispatcherTimer(DispatcherPriority.Input, _dispatcher)
        {
            Interval = PushToTalkPollInterval,
        };

        // The chord is still down at the moment the hotkey message arrives, but a press short
        // enough to be released before the first tick would otherwise stop instantly. Waiting for
        // one observed "held" reading makes a tap behave like a very short hold rather than a
        // no-op.
        var everHeld = false;
        var idleTicks = 0;
        _pushToTalkTimer.Tick += (_, _) =>
        {
            if (_isHotKeyHeld())
            {
                everHeld = true;
                return;
            }

            // A press released before the first tick is never observed as held. Stopping after a
            // few idle ticks anyway means the worst case is a tap that listens for a moment, not a
            // session that never ends.
            if (everHeld || ++idleTicks >= 3)
                Stop();
        };
        _pushToTalkTimer.Start();
    }

    private void StopPushToTalkPolling()
    {
        _pushToTalkTimer?.Stop();
        _pushToTalkTimer = null;
    }
}
