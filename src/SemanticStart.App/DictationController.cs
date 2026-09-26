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

        // The start gate is deliberately not disposed: a start that is still between its wait and
        // its release would then throw on a background task during shutdown, which is a crash log
        // for no benefit. Letting it be collected costs nothing.

        // The warm-start task owns the engine until it completes, so disposing here would race it.
        // Hand the disposal to whichever finishes last.
        if (_warmStart is { } warmStart)
            _ = warmStart.ContinueWith(t => t.Result?.Dispose(), TaskContinuationOptions.OnlyOnRanToCompletion);
        else
            _engine?.Dispose();
    }

    private async Task<DictationEngine?> CreateEngineAsync()
    {
        try
        {
            var models = new SpeechModelBootstrapper();
            var selector = new SpeechEngineSelector([new SherpaOnnxSpeechProvider(models)]);
            var startup = await selector.StartAsync().ConfigureAwait(false);

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
            Log.Info($"Dictation ready: {engine.Metadata.DisplayName}.");
            return engine;
        }
        catch (Exception ex)
        {
            _unavailableReason = "The speech model could not be loaded; see the log for details.";
            Log.Error(ex, "Failed to start the dictation engine");
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
