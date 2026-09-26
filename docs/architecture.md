# Solution architecture

A map of the whole solution, for orientation during review. The dictation feature has its own
companion document: [`architecture-speech.md`](architecture-speech.md).

For the product-level explanation of the pipeline — why hybrid retrieval, which collectors, which
enrichers — see the **How it works** section of the [README](../README.md). This document is about
where the code lives and what calls what.

## Projects

```mermaid
flowchart TB
    Core["SemanticStart.Core<br/><i>library — no UI</i>"]
    App["SemanticStart.App<br/><i>WPF overlay, tray, --mcp mode</i>"]
    Cli["SemanticStart.Cli<br/><i>index, search, eval, stats, enrich, diagnose</i>"]
    Tests["SemanticStart.Tests"]
    Probe1["benchmarks/DictationLatencyProbe"]
    Probe2["benchmarks/TensorPrimitivesProbe"]

    App --> Core
    Cli --> Core
    Tests --> Core
    Tests --> App
    Probe1 -.-> Core

    classDef outside stroke-dasharray: 4 4
    class Probe1,Probe2 outside
```

Dashed projects are not in `SemanticStart.slnx` — they are run by hand on real hardware and are
deliberately kept out of the build. `Core` never references `App`: the query engine is a standalone
library with no UI dependency, so it can back a Command Palette or PowerToys Run extension later.

## Core, by folder

```mermaid
flowchart LR
    Abstractions["Abstractions<br/><i>IEntityCollector, IEnricher,<br/>ISearchEngine, IIndexStore,<br/>IEntityLauncher</i>"]
    Model["Model<br/><i>Entity, EnrichmentDocument,<br/>SynthesizedProfile, SearchHit</i>"]
    Collectors["Collectors<br/><i>10 collectors, in dedup order</i>"]
    Enrichment["Enrichment<br/><i>EnrichmentPipeline + enrichers</i>"]
    Synthesis["Synthesis<br/><i>HeuristicProfileSynthesizer</i>"]
    Embeddings["Embeddings<br/><i>OnnxEmbeddingGenerator,<br/>EmbeddingModelBootstrapper</i>"]
    Indexing["Indexing<br/><i>IndexBuilder</i>"]
    Storage["Storage<br/><i>SqliteIndexStore, VectorFile</i>"]
    Query["Query<br/><i>HybridSearchEngine,<br/>SemanticIndexRuntime</i>"]
    Launching["Launching<br/><i>ShellEntityLauncher</i>"]
    Speech["Speech<br/><i>dictation — see companion doc</i>"]

    Collectors --> Model
    Enrichment --> Model
    Synthesis --> Model
    Indexing --> Collectors
    Indexing --> Enrichment
    Indexing --> Synthesis
    Indexing --> Embeddings
    Indexing --> Storage
    Query --> Storage
    Query --> Embeddings
    Launching --> Model
    Storage --> Model
    Collectors -.-> Abstractions
    Enrichment -.-> Abstractions
    Storage -.-> Abstractions
    Query -.-> Abstractions
    Launching -.-> Abstractions

    classDef speech fill:#eef5ff,stroke:#4a76c4,color:#102a43
    class Speech speech
```

`Speech` sits apart on purpose: nothing else in Core depends on it, and it depends on nothing in
Core but `AppPaths`. It shares only the ONNX Runtime with `Embeddings` — which is the one real
coupling, and the reason the two native package versions must move together.

## Indexing — offline

```mermaid
flowchart LR
    subgraph collect["Discover"]
        C1["AppsFolderCollector"]
        C2["...8 more..."]
        C3["PathExecutableCollector"]
    end
    Diff{"ContentHash<br/>changed?"}
    Enrich["EnrichmentPipeline<br/><i>local + optional online</i>"]
    Synth["HeuristicProfileSynthesizer"]
    Embed["OnnxEmbeddingGenerator<br/><i>all-MiniLM-L6-v2, 384-d</i>"]
    Store[("index.sqlite<br/>+ vectors.bin")]

    C1 --> Diff
    C2 --> Diff
    C3 --> Diff
    Diff -->|"new or changed"| Enrich --> Synth --> Embed --> Store
    Diff -->|"unchanged"| Store
```

Collector order is deduplication precedence: an earlier source wins when the same thing is found
twice, which is why `PathExecutableCollector` runs last. `IndexBuilder` drives the whole sequence
and reports through `IProgress<IndexProgress>`.

## Querying — the hot path

```mermaid
flowchart TB
    Query["query text"]
    Vector["Vector arm<br/><i>cosine over vectors.bin,<br/>TensorPrimitives SIMD</i>"]
    Lexical["Lexical arm<br/><i>FTS5 / BM25</i>"]
    Fuse["Reciprocal Rank Fusion<br/><i>+ literal-name, adjacency, usage boosts</i>"]
    Floors["Surface floors<br/><i>weak evidence dropped, not padded</i>"]
    Hits["SearchHit[]"]
    Launch["ShellEntityLauncher<br/><i>+ RecordLaunchAsync</i>"]

    Query --> Vector --> Fuse
    Query --> Lexical --> Fuse
    Fuse --> Floors --> Hits --> Launch
    Launch -.->|"usage feeds ranking"| Fuse
```

`HybridSearchEngine` owns fusion and ranking; `SemanticIndexRuntime` is the read-only, query-only
entry point used by both the overlay and the MCP server.

## The app

```mermaid
flowchart TB
    subgraph startup["App.OnStartup — composition root"]
        direction TB
        S1["Log.Initialize + crash handlers"]
        S2["SpeechTelemetry.Install"]
        S3["single-instance mutex"]
        S4["AppSettingsService → AppSettings"]
        S5["SemanticSearchService"]
        S6["OverlayViewModel + OverlayWindow"]
        S7["DictationController"]
        S8["ActivationManager"]
        S9["TrayIconService"]
        S10["IndexRebuildCoordinator + IndexRefreshScheduler"]
        S11["warm start: index + recogniser"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10 --> S11
    end
```

There is no DI container here. `App.xaml.cs` is a manual composition root: it constructs each
service in order and disposes it in `OnExit`. The one exception is `McpServerHost`, which uses
`Host.CreateApplicationBuilder` because the MCP SDK requires it.

```mermaid
flowchart LR
    Hotkey["ActivationManager<br/><i>RegisterHotKey, two ids</i>"]
    Tray["TrayIconService"]
    Overlay["OverlayWindow"]
    VM["OverlayViewModel"]
    Debounce["SearchDebouncer"]
    Search["SemanticSearchService"]
    Runtime["SemanticIndexRuntime"]
    Dictation["DictationController"]
    Settings["SettingsWindow"]
    Rebuild["IndexRebuildCoordinator"]
    Refresh["IndexRefreshScheduler"]
    Mcp["McpServerHost<br/><i>--mcp</i>"]

    Hotkey -->|"activation chord"| Overlay
    Hotkey -->|"dictation chord"| Dictation
    Tray --> Overlay
    Tray --> Settings
    Tray --> Rebuild
    Overlay --> VM
    Dictation -->|"Query only"| VM
    VM --> Debounce --> Search --> Runtime
    Settings --> Dictation
    Settings --> Rebuild
    Refresh --> Rebuild
    Mcp --> Runtime

    classDef speech fill:#eef5ff,stroke:#4a76c4,color:#102a43
    class Dictation speech
```

The arrow to watch in review is `DictationController → OverlayViewModel`. It is the *only* edge
this PR adds into the existing query path, and it sets `Query` and nothing else — partials through
the ordinary setter so `SearchDebouncer` decides when to search, finals flushing it.

## On disk

All under `%LOCALAPPDATA%\SemanticStart` (`AppPaths.Root`, overridable with `SEMANTICSTART_HOME`):

| Path | Contents |
|---|---|
| `index.sqlite` | Entities, enrichment documents, profiles, FTS5 mirror, usage stats |
| `vectors.bin` | L2-normalized 384-d embeddings, row-major |
| `models/` | `all-MiniLM-L6-v2` for the index; the Zipformer and Silero models for dictation |
| `cache/enrichment/`, `cache/icons/` | Cached online payloads and extracted shell icons |
| `settings.json` | `AppSettings` |
| `logs/app.log` | The single log, `INFO` / `TRACE` / `ERROR` lines |

## Observability

`SemanticStart.Core` does not log. It reports progress through `IProgress<T>` and failure through
exceptions, and the app decides what to write. Dictation keeps that rule: it emits
`System.Diagnostics.Activity` spans on the `SemanticStart.Speech` source and
`SpeechDiagnostics.Reported` events for anything it handles internally, and `SpeechTelemetry` — in
the app, next to `Log` — is the single subscriber that turns those into lines in `app.log`.

```mermaid
flowchart LR
    CoreCode["Core.Speech"] -->|"ActivitySource"| Listener["SpeechTelemetry"]
    CoreCode -->|"Reported events"| Listener
    Listener --> Log["Log → app.log"]
    CoreCode -.->|"same source, no app change"| External["dotnet-trace /<br/>OpenTelemetry"]
    AppCode["App services"] --> Log
```

Spans carry durations and lengths only; transcript text is never written anywhere.
