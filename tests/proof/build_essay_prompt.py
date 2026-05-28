#!/usr/bin/env python3
"""Deterministic generator for tests/long_context_essay_prompt.txt.

The essay prompt is the long-response counterpart to
`tests/long_context_story_prompt.txt`.  The story prompt asks the model to
recall a few numbers from a long context, so it produces short output and is
useless for testing decode parity past a few dozen tokens.  This fixture asks
for an extremely long multi-chapter essay; with `-n 1024` the model fills the
budget without natural EOS.

The body is built from a static topics table so re-running this script with
no arguments produces a byte-identical fixture.  Edit the topics table if you
need different content; do NOT add randomness, dates, or any other source of
nondeterminism.

Usage:
  python3 tests/proof/build_essay_prompt.py            # write to default path
  python3 tests/proof/build_essay_prompt.py --check    # diff against on-disk file

The default output path is `tests/long_context_essay_prompt.txt`, resolved
relative to the repo root (two directories above this script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Each topic contributes a heading and four short prose blocks.  At ~85 words
# per block, 24 topics gives ~8200 words of body, which tokenizes to roughly
# 10k tokens with the DeepSeek tokenizer -- enough to push the model well past
# the pos~10k regime where the substrate-driven bug class manifests.  All text
# is hand-written analytical prose; no LLM-generated content.
TOPICS: list[dict[str, str]] = [
    {
        "title": "Modern Processor Pipelines",
        "background": (
            "A contemporary out-of-order processor decodes instructions into "
            "micro-operations, renames their registers to remove false "
            "dependencies, queues them in a reservation station, and then "
            "dispatches them to execution units as their operands become "
            "available. Retirement is in original program order to preserve "
            "the illusion of sequential execution."
        ),
        "mechanism": (
            "Reorder buffers track each in-flight instruction's status; load "
            "queues and store queues let memory operations reorder while "
            "preserving single-thread consistency. Branch predictors steer "
            "the front end past control-flow uncertainty so the back end can "
            "keep its dozens of execution units fed."
        ),
        "tradeoff": (
            "Wider issue width raises peak throughput but also raises misprediction "
            "cost, since more in-flight work has to be squashed when speculation "
            "loses. Designers balance window size, scheduler latency, and "
            "predictor accuracy against the energy cost of dynamic scheduling."
        ),
        "practice": (
            "Software that exploits this hardware exposes parallelism: "
            "independent dependency chains, predictable branches, and locality. "
            "Profile-guided layouts let the compiler co-design with the front "
            "end, hoisting hot blocks and aligning loops so the predictor sees "
            "consistent targets."
        ),
    },
    {
        "title": "Cache Hierarchy and Locality",
        "background": (
            "Modern systems hide DRAM latency with a hierarchy: small fast L1 "
            "caches per core, a larger shared L2, and a still larger L3 "
            "spanning a socket. Each level trades capacity for latency, and "
            "each enforces a particular coherence protocol with the others."
        ),
        "mechanism": (
            "Line fills move 64 or 128 bytes at a time, so spatial locality "
            "amortizes the cost. Temporal locality keeps a working set resident; "
            "prefetchers detect strided patterns and issue speculative loads to "
            "stay ahead of demand."
        ),
        "tradeoff": (
            "Larger caches lower miss rates but raise access latency and area. "
            "Replacement policies trade implementation cost against fairness "
            "between hot and cold lines. Coherence traffic on writes scales "
            "with sharing, which is why scalable algorithms partition state."
        ),
        "practice": (
            "Structure-of-arrays layouts, cache-line-padded counters, and "
            "thread-local accumulators are recurring patterns. Engineers "
            "measure miss rates and false-sharing rates as routinely as they "
            "measure throughput, because a cache mistake easily costs an order "
            "of magnitude."
        ),
    },
    {
        "title": "Memory Models",
        "background": (
            "A memory model specifies what writes a thread is guaranteed to "
            "observe from other threads and in what order. Stronger models "
            "are easier to reason about; weaker models leave more freedom for "
            "hardware and compilers to reorder operations."
        ),
        "mechanism": (
            "Acquire-release semantics anchor ordering at synchronization "
            "points. Atomics provide explicit ordering hints so the compiler "
            "and the hardware know which reorderings are still allowed. Fences "
            "force a global ordering boundary when nothing weaker suffices."
        ),
        "tradeoff": (
            "Sequentially consistent code is easy to reason about but expensive "
            "to implement; relaxed code is fast but error-prone. Most "
            "production code lands at acquire-release, which is the sweet "
            "spot for lock-free data structures."
        ),
        "practice": (
            "Reviewers look for missing acquire or release annotations on "
            "shared state, for relaxed atomics paired with manual fences, and "
            "for spinlocks that forgot to release on the failure path. "
            "Test harnesses use stress and randomized scheduling to surface "
            "model violations."
        ),
    },
    {
        "title": "Floating-Point and Quantization",
        "background": (
            "Floating-point numbers approximate the reals with finite "
            "precision, so reductions over different orderings produce "
            "different results. Lower precisions (FP16, BF16, FP8) trade "
            "bits for throughput and memory bandwidth."
        ),
        "mechanism": (
            "Quantized inference packs weights into a few bits per value, "
            "stores per-block scales, and reconstructs approximate floats "
            "at compute time. Accumulators keep higher precision so the "
            "sum across many rows does not underflow."
        ),
        "tradeoff": (
            "Aggressive quantization shrinks models and increases throughput "
            "but degrades quality once the bit budget drops below the natural "
            "dynamic range of the weights. Mixed precision per layer is the "
            "norm because attention and feed-forward have different "
            "sensitivities."
        ),
        "practice": (
            "Calibration sets choose the scales; outlier-aware methods preserve "
            "the long tail that drives end-task quality. Determinism work "
            "requires picking one reduction order and sticking to it; FP "
            "noise is real noise, not a bug, but it is often confused for one."
        ),
    },
    {
        "title": "GPU Execution Model",
        "background": (
            "A modern GPU runs thousands of threads grouped into warps that "
            "execute in lockstep on a single SIMD lane. Many warps are "
            "co-resident on each streaming multiprocessor so memory latency "
            "is overlapped with computation from independent warps."
        ),
        "mechanism": (
            "Kernels declare grids of blocks; blocks share fast on-chip memory "
            "and synchronize among their threads. Global memory accesses "
            "coalesce when consecutive lanes hit consecutive addresses; "
            "scattered accesses tank effective bandwidth."
        ),
        "tradeoff": (
            "Higher occupancy hides latency but reduces per-thread register "
            "budget. Shared-memory tiling raises arithmetic intensity but "
            "complicates the kernel. Designers weigh launch overhead against "
            "the cost of running too few thread blocks."
        ),
        "practice": (
            "Profilers report memory throughput, occupancy, and warp stalls. "
            "Persistent kernels and graph capture amortize launch cost; "
            "tensor cores raise arithmetic density when the layout allows. "
            "Decoder-side inference work has tiny batches, so launch overhead "
            "dominates."
        ),
    },
    {
        "title": "Kernel Graph Capture and Replay",
        "background": (
            "Each kernel launch carries a fixed overhead. For small per-step "
            "decode kernels this overhead dominates wall time. Graph capture "
            "records a sequence of launches once and replays the whole graph "
            "with a single submission afterwards."
        ),
        "mechanism": (
            "The runtime traces dependencies between captured launches and "
            "submits them as a single batch. Replay skips driver-side "
            "validation, argument copy, and stream-state synchronization, "
            "saving microseconds that add up across hundreds of layers per "
            "decoded token."
        ),
        "tradeoff": (
            "Captured arguments are frozen at capture time. Any kernel value "
            "that changes between decodes -- token id, KV row count, attention "
            "window boundary -- must be sourced from a live device-side "
            "substrate, not passed by value. Forgetting this fact is the "
            "source of a recurring bug class."
        ),
        "practice": (
            "Per-layer graph caches key on observable regime changes so the "
            "right graph variant is picked at replay. Hash dumps of per-kernel "
            "outputs let you bisect divergences between eager and captured "
            "runs without scaffolding a separate test rig."
        ),
    },
    {
        "title": "Speculative Decoding",
        "background": (
            "Speculative decoding accelerates autoregressive generation by "
            "drafting several future tokens with a cheap predictor, then "
            "verifying them with the expensive base model in a single forward "
            "pass. Verified tokens commit; rejected ones are dropped."
        ),
        "mechanism": (
            "The draft model is small enough that its per-token cost is a "
            "fraction of the base model. Verification reuses the base model's "
            "output distribution to accept or reject each draft token in "
            "sequence, so the worst case is no slower than non-speculative "
            "decoding."
        ),
        "tradeoff": (
            "High acceptance rate yields large speedups; low acceptance wastes "
            "draft compute. Draft quality, draft depth, and verification "
            "policy all trade against each other. Exact replay paths preserve "
            "argmax equivalence at the cost of some throughput."
        ),
        "practice": (
            "Verifiers must respect the same numerical regime as the unaided "
            "decoder, or accepted tokens will diverge from the non-speculative "
            "reference. Acceptance traces and per-step timing decompositions "
            "let engineers attribute gains correctly to the draft and the "
            "verifier."
        ),
    },
    {
        "title": "Attention Indexers and Sparse Decoders",
        "background": (
            "Attention over the full context grows quadratically, so long "
            "contexts demand either approximate or sparse attention. Indexer "
            "kernels select a top-k subset of past tokens for each new query "
            "and run attention against that subset."
        ),
        "mechanism": (
            "An indexer typically projects keys into a low-rank space, scores "
            "the candidates, and emits the top-k indices for the attention "
            "kernel to consume. Compressed key-value caches further reduce "
            "memory bandwidth on the read path."
        ),
        "tradeoff": (
            "Aggressive sparsity loses signal from rarely-selected tokens; "
            "too-loose indexing wastes attention compute. Compression bits "
            "trade memory bandwidth against arithmetic precision; predecoded "
            "scales avoid repeated reconstruction at the cost of extra "
            "scratch."
        ),
        "practice": (
            "Indexed paths have their own captured-decode invariants: the "
            "top-k count, the compressed row count, and the indexer query "
            "key all depend on live decode state. By-value parameters that "
            "freeze at capture time produce subtle long-context regressions."
        ),
    },
    {
        "title": "Distributed Consensus",
        "background": (
            "Replicated systems use consensus protocols (Paxos, Raft, "
            "Zab, viewstamped replication) to agree on the order of "
            "operations across nodes despite delays, partial failures, and "
            "asynchronous networks."
        ),
        "mechanism": (
            "A leader proposes; a quorum of followers acknowledges; a "
            "decision becomes durable once the quorum is reached. Leaders "
            "rotate by election, with fencing and term numbers keeping a "
            "deposed leader from continuing to write."
        ),
        "tradeoff": (
            "Stronger consistency lowers throughput and raises latency. "
            "Quorum size trades availability against safety. Replicated "
            "state machines simplify reasoning but require deterministic "
            "log apply, which constrains how state is computed."
        ),
        "practice": (
            "Engineers test these systems with deterministic simulators, "
            "fault injection, and randomized message reordering. The hardest "
            "bugs hide in the transitions between configurations or between "
            "term numbers, not in steady-state operation."
        ),
    },
    {
        "title": "Database Storage Engines",
        "background": (
            "A storage engine persists rows or key-value pairs to durable "
            "media and exposes lookup, scan, insert, update, and delete. The "
            "two dominant families are B-tree-based and log-structured."
        ),
        "mechanism": (
            "B-trees keep an in-place sorted structure and rewrite pages on "
            "update. Log-structured stores append all writes and compact "
            "older data in background. Both employ caching, write-ahead "
            "logging, and crash-recovery routines."
        ),
        "tradeoff": (
            "B-trees deliver predictable read amplification; log-structured "
            "engines deliver lower write amplification. Compaction overhead "
            "shifts work in time but does not eliminate it. Hybrid designs "
            "borrow from both families to suit the access pattern."
        ),
        "practice": (
            "Benchmarks measure point-read latency, scan throughput, write "
            "amplification, and tail latency. Production tuning chases "
            "compaction stalls, cold-cache reads, and the long tail of "
            "transaction latency at the 99.9th percentile."
        ),
    },
    {
        "title": "Concurrency Primitives",
        "background": (
            "Threads coordinate through mutexes, condition variables, "
            "semaphores, channels, and lock-free data structures. Each "
            "primitive trades some combination of contention behavior, "
            "fairness, and programming difficulty."
        ),
        "mechanism": (
            "A mutex serializes access; a condition variable parks waiters "
            "until a predicate becomes true; a channel hands ownership of a "
            "datum between threads. Lock-free structures rely on atomic "
            "compare-and-swap to make progress without holding any lock."
        ),
        "tradeoff": (
            "Coarse locks are simple but limit throughput; fine-grained "
            "locks scale better but invite deadlocks. Lock-free code avoids "
            "blocking but requires meticulous attention to memory ordering, "
            "ABA hazards, and reclamation."
        ),
        "practice": (
            "Code reviews catch the obvious mistakes; stress tests, runtime "
            "race detectors, and model checkers catch the rest. Production "
            "incidents around concurrency disproportionately come from "
            "accidental sharing of supposedly thread-local state."
        ),
    },
    {
        "title": "Compiler Optimization",
        "background": (
            "Compilers translate human-friendly source into machine code, "
            "applying transformations that preserve observable behavior "
            "while improving speed or size. Optimization passes work over "
            "an intermediate representation that exposes data flow and "
            "control flow."
        ),
        "mechanism": (
            "Classical passes include constant folding, common-subexpression "
            "elimination, loop-invariant code motion, vectorization, and "
            "inlining. Modern passes also reason about aliasing, memory "
            "models, and undefined behavior."
        ),
        "tradeoff": (
            "Aggressive inlining grows code size and can hurt instruction "
            "cache behavior; aggressive vectorization can complicate "
            "alignment requirements. Profile-guided optimization narrows "
            "the choice space to what actually pays off in production."
        ),
        "practice": (
            "Engineers read disassembly when performance is critical, write "
            "no-op shims to defeat unsafe constant propagation, and pin "
            "compiler versions because optimization heuristics drift across "
            "releases."
        ),
    },
    {
        "title": "Filesystems",
        "background": (
            "A filesystem organizes durable storage into files and "
            "directories. It tracks metadata (ownership, permissions, "
            "timestamps), maps logical offsets to physical blocks, and "
            "provides crash consistency guarantees that applications can "
            "build on."
        ),
        "mechanism": (
            "Journals or copy-on-write structures keep metadata consistent "
            "across crashes. Block allocation policies, extent maps, and "
            "delayed allocation reduce fragmentation. Page caches hide "
            "media latency from applications."
        ),
        "tradeoff": (
            "Synchronous writes are durable but slow; asynchronous writes are "
            "fast but lose recent data on a crash. Snapshots are cheap on "
            "copy-on-write filesystems and expensive on in-place ones. "
            "Inline encryption and checksumming cost CPU but harden the data."
        ),
        "practice": (
            "Production checklists pin mount options, verify fsync semantics, "
            "and exercise crash-and-recover scenarios. Long-running services "
            "discover the unhappy paths first; tests must cover them up "
            "front."
        ),
    },
    {
        "title": "Networking Stacks",
        "background": (
            "Network stacks layer functionality: physical, link, network, "
            "transport, application. Each layer encapsulates the next and "
            "addresses a different concern, from electrical signaling to "
            "ordered byte streams to higher-level protocols."
        ),
        "mechanism": (
            "TCP provides ordered, reliable, congestion-controlled byte "
            "streams; UDP provides best-effort datagrams. Userspace stacks "
            "and kernel bypass mechanisms trade portability for raw "
            "throughput. Multi-queue NICs spread interrupt load across "
            "cores."
        ),
        "tradeoff": (
            "Reliability and ordering cost latency. Congestion control "
            "smooths the network at the cost of peak throughput. Encryption "
            "costs CPU; offloads recover some of it on capable hardware."
        ),
        "practice": (
            "Engineers tune queue depths, NIC offloads, and CPU affinity; "
            "they also profile across user-kernel boundaries. Network-bound "
            "services live or die by the tail of the latency distribution, "
            "not the mean."
        ),
    },
    {
        "title": "Build Systems",
        "background": (
            "A build system reproduces software artifacts from source, "
            "tracking dependencies and caching intermediate outputs. The "
            "field ranges from simple recursive Makefiles to sophisticated "
            "content-addressed graph evaluators."
        ),
        "mechanism": (
            "Each rule declares inputs, outputs, and a command. The build "
            "system compares timestamps or content hashes to decide what "
            "must rerun. Remote caching shares outputs across machines so "
            "a fresh checkout is not a full rebuild."
        ),
        "tradeoff": (
            "Strict dependency tracking enables aggressive caching but raises "
            "the cost of adding new rules. Hermetic builds reproduce across "
            "machines but require explicit dependency declarations everywhere, "
            "including for system headers and toolchains."
        ),
        "practice": (
            "Teams audit build times routinely, because slow builds erode "
            "developer iteration speed. CI caches and incremental local "
            "rebuilds are the two levers; both depend on accurate dependency "
            "tracking."
        ),
    },
    {
        "title": "Continuous Integration",
        "background": (
            "Continuous integration runs the test suite on every change "
            "before it lands, so the main branch stays green. Build and "
            "test infrastructure scales to the team's commit rate; flaky "
            "tests are quickly bisected and either fixed or quarantined."
        ),
        "mechanism": (
            "Hooks trigger pipelines on push or pull-request events. "
            "Pipelines build, lint, and test in parallel stages; cache "
            "layers reuse compiled outputs across runs. Results post "
            "back to the change as status checks."
        ),
        "tradeoff": (
            "Faster pipelines need more parallelism and more cache hits, "
            "which costs infrastructure. Pre-merge enforcement keeps the "
            "main branch usable at the cost of merge throughput; "
            "post-merge enforcement risks regressions slipping in."
        ),
        "practice": (
            "Engineering organizations treat CI as a first-class product, "
            "tracking flake rates, queue depth, and end-to-end latency. "
            "Test selection and ordering shrink the per-change cost without "
            "sacrificing coverage."
        ),
    },
    {
        "title": "Garbage Collection",
        "background": (
            "Managed runtimes reclaim unreachable memory automatically, "
            "freeing programmers from manual lifetimes but introducing a "
            "scheduler that competes with application threads for CPU and "
            "memory bandwidth."
        ),
        "mechanism": (
            "Generational collectors exploit the observation that most "
            "objects die young; concurrent collectors interleave work with "
            "the application; region-based allocators reduce pause times "
            "by promoting older data to separate regions."
        ),
        "tradeoff": (
            "Lower pause times require more concurrent work and tighter "
            "barriers. Higher throughput collectors pause longer. Memory "
            "headroom always trades against frequency: a tight heap collects "
            "often, a loose heap collects rarely but uses more RAM."
        ),
        "practice": (
            "Latency-sensitive services tune the collector for predictable "
            "pause budgets, sometimes at significant throughput cost. "
            "Allocation-rate profiling pinpoints the hot paths that drive "
            "collection frequency in the first place."
        ),
    },
    {
        "title": "Type Systems",
        "background": (
            "A type system classifies values and rejects programs that "
            "would misuse them. Static type checkers run before execution; "
            "dynamic ones check at runtime. Each style trades coverage "
            "against flexibility."
        ),
        "mechanism": (
            "Hindley-Milner inference assigns types without annotations in "
            "many cases. Algebraic data types express disjoint unions "
            "exhaustively. Effect types track side effects in the type "
            "system, surfacing latency and resource concerns at compile "
            "time."
        ),
        "tradeoff": (
            "Stronger types raise upfront cost and reduce expressiveness "
            "for valid programs that happen to be hard to type. They lower "
            "the runtime cost of defensive code and the residual cost of "
            "production bugs that hard checks would have caught."
        ),
        "practice": (
            "Teams that adopt strict typing usually invest in IDE tooling, "
            "code generators, and migration guides at the same time. The "
            "type system is only as valuable as the tools that make it cheap "
            "to use."
        ),
    },
    {
        "title": "Profiling and Observability",
        "background": (
            "Production systems emit metrics, logs, and traces to expose "
            "behavior in real time. Sampling profilers attribute CPU time "
            "across functions; tracing tools follow individual requests "
            "across services."
        ),
        "mechanism": (
            "Counters and histograms expose aggregate behavior cheaply; "
            "traces capture causal chains at the cost of more overhead. "
            "Sampling balances coverage against ingestion cost. Structured "
            "logs become queryable as JSON or as columnar records."
        ),
        "tradeoff": (
            "Higher observability raises operational cost and storage "
            "footprint. Cardinality explosions in labels overwhelm "
            "back-ends. Useful traces span service boundaries, so "
            "propagation must be reliable and lightweight."
        ),
        "practice": (
            "Site reliability teams use observability to debug fast and "
            "to validate incident response. They invest in tooling that "
            "summarizes signals automatically, because dashboards "
            "alone do not scale to large fleets."
        ),
    },
    {
        "title": "Software Architecture Patterns",
        "background": (
            "Architectural patterns offer common solutions: layered, "
            "hexagonal, event-driven, microservices, modular monolith. "
            "Each emerged from a class of operational pressure and works "
            "best when those pressures still apply."
        ),
        "mechanism": (
            "Layered designs separate presentation, application, and "
            "domain logic. Hexagonal designs invert dependencies so the "
            "domain does not import infrastructure. Event-driven systems "
            "decouple producers from consumers, at the cost of asynchronous "
            "reasoning."
        ),
        "tradeoff": (
            "Microservices enable independent deployment but require "
            "investment in coordination, observability, and platform "
            "engineering. Modular monoliths preserve simpler operations "
            "but require discipline to keep modules decoupled. Premature "
            "decomposition is a frequent failure mode."
        ),
        "practice": (
            "Architects revisit patterns as teams grow and as the system "
            "matures. They look for friction in deployment, in "
            "ownership boundaries, and in cross-team coordination, not "
            "for theoretical purity."
        ),
    },
    {
        "title": "Testing Strategies",
        "background": (
            "A healthy test suite contains unit, integration, end-to-end, "
            "property, and performance tests in proportions that match the "
            "system's risk profile. Each layer answers a different question "
            "about correctness."
        ),
        "mechanism": (
            "Unit tests pin specific functions; integration tests verify "
            "module boundaries; end-to-end tests exercise full flows. "
            "Property tests assert invariants over random inputs. Fuzzers "
            "find inputs nobody else thought of."
        ),
        "tradeoff": (
            "Heavier tests catch more but cost more to maintain and to "
            "run. Mocks accelerate tests but drift from reality. "
            "Production-data replay finds real regressions at the cost "
            "of complex fixtures and data privacy concerns."
        ),
        "practice": (
            "Mature teams budget for test-suite latency and reliability "
            "explicitly. Coverage is a useful proxy when it grows; it is "
            "misleading when it stagnates. Mutation testing exposes "
            "weak assertions."
        ),
    },
    {
        "title": "Code Review",
        "background": (
            "Peer review of changes catches defects, spreads knowledge, "
            "and improves design. It is one of the cheapest interventions "
            "in software quality and one of the easiest to do badly."
        ),
        "mechanism": (
            "Reviewers read the diff, often the surrounding context, run "
            "tests locally for risky changes, and leave inline comments. "
            "Authors revise; the cycle continues until both sides are "
            "satisfied. Automated checks reduce reviewer workload on "
            "mechanical concerns."
        ),
        "tradeoff": (
            "Heavier review raises change latency and merge-throughput "
            "cost. Lighter review accelerates work but admits more bugs. "
            "Asynchronous review scales but loses the bandwidth of "
            "in-person discussion."
        ),
        "practice": (
            "Effective reviewers focus on design, edge cases, and intent. "
            "They tag the riskiest paths for fresh eyes; they accept that "
            "style is a solved problem deferred to tooling. They ask "
            "questions before recommending changes."
        ),
    },
    {
        "title": "Documentation",
        "background": (
            "Working software needs three documents: an explanation of "
            "what it does, a guide to operating it, and a reference for "
            "extending it. Each audience reads a different one; collapsing "
            "them confuses everyone."
        ),
        "mechanism": (
            "Generated reference docs stay in sync with code by living next "
            "to it. Hand-written explanations record intent that the code "
            "alone cannot. Runbooks operationalize what worked when an "
            "incident hit and what did not."
        ),
        "tradeoff": (
            "Documentation that is too detailed rots quickly; documentation "
            "that is too sparse is useless. Teams who pay the cost of "
            "writing docs reap leverage in onboarding; teams who skip it "
            "rebuild the same context every quarter."
        ),
        "practice": (
            "Engineering organizations periodically prune stale docs as "
            "diligently as they add new ones. A doc with the wrong steps "
            "is worse than no doc; honest 'last verified' notes are a "
            "cheap signal."
        ),
    },
    {
        "title": "Release Engineering",
        "background": (
            "Shipping software end-to-end requires more than a green test "
            "suite. Release engineering covers versioning, change logs, "
            "compatibility surfaces, deprecation policies, and the human "
            "coordination around each step."
        ),
        "mechanism": (
            "Semantic versioning communicates compatibility expectations. "
            "Feature flags decouple deploy from release. Progressive "
            "rollouts (canary, blue/green, percent-based) catch problems "
            "before the full population is exposed."
        ),
        "tradeoff": (
            "Slow rollouts catch regressions but stretch the time between "
            "merge and audience. Fast rollouts compress that gap but raise "
            "blast radius on bad changes. Most teams settle somewhere in "
            "between based on the cost of a regression in their domain."
        ),
        "practice": (
            "Release dashboards make the state of every artifact visible. "
            "Rollback procedures are exercised so they are real; "
            "communication conventions ensure the right humans know when "
            "the right things ship."
        ),
    },
    {
        "title": "Containerization and Isolation",
        "background": (
            "Containers package an application together with its dependencies "
            "into an immutable image that runs identically on any host that "
            "speaks the same kernel ABI. They provide process-level isolation "
            "without the resource cost of full virtualization."
        ),
        "mechanism": (
            "Linux namespaces wall off process trees, mounts, networks, and "
            "user identifiers. Control groups account and cap resource usage "
            "per container. Layered image formats let unchanged base layers "
            "share bytes across many derived images and pull incrementally."
        ),
        "tradeoff": (
            "Containers share the host kernel, so a kernel vulnerability "
            "can defeat isolation. They start faster and consume less "
            "memory than virtual machines but offer weaker security "
            "guarantees in adversarial multi-tenant settings."
        ),
        "practice": (
            "Production platforms run containers under orchestrators that "
            "schedule them onto hosts, mount their secrets, route their "
            "traffic, and restart them when they fail. The orchestrator is "
            "where most of the operational complexity now lives."
        ),
    },
    {
        "title": "Service Mesh and Sidecars",
        "background": (
            "A service mesh handles cross-cutting concerns -- mutual "
            "authentication, load balancing, retries, circuit breaking, "
            "observability -- in a sidecar process that runs next to each "
            "application instance and intercepts its traffic."
        ),
        "mechanism": (
            "Each pod or VM runs an application container and an envoy-like "
            "sidecar; a control plane configures the sidecars centrally. "
            "Policy is expressed declaratively and propagates to the data "
            "plane without changes to application code."
        ),
        "tradeoff": (
            "Sidecars add latency and resource overhead on every hop and "
            "double the number of moving parts per service. They are "
            "justified when many services need consistent policy that "
            "application teams should not reimplement individually."
        ),
        "practice": (
            "Adoption pays off when the platform team owns the mesh and the "
            "application teams pay no daily cost. It pays poorly when each "
            "service team must understand the mesh configuration to ship "
            "their next change."
        ),
    },
    {
        "title": "Authentication and Authorization",
        "background": (
            "Authentication establishes who is making a request; "
            "authorization decides whether that requester is permitted to "
            "perform the operation. Modern systems separate the two and "
            "compose them through tokens, claims, and policies."
        ),
        "mechanism": (
            "OAuth flows mint short-lived tokens with embedded claims; "
            "policy engines evaluate those claims against rules expressed "
            "in a small declarative language. Audit logs record the "
            "decisions for after-the-fact review and incident response."
        ),
        "tradeoff": (
            "Short token lifetimes contain blast radius from theft but "
            "raise refresh-traffic load. Centralized policy is auditable "
            "but introduces a hot dependency on the policy service. "
            "Coarse policies underbind; fine policies are hard to manage."
        ),
        "practice": (
            "Security teams favor least-privilege defaults, periodic access "
            "reviews, and break-glass procedures with mandatory after-the-fact "
            "justification. Every system grows quietly toward over-permissive "
            "if no countervailing pressure exists."
        ),
    },
    {
        "title": "Cryptographic Primitives",
        "background": (
            "Cryptographic systems rest on a small set of primitives: "
            "symmetric ciphers, public-key cryptosystems, hash functions, "
            "key derivation, and authenticated encryption. Each has a "
            "well-defined contract and a body of attacks that motivated it."
        ),
        "mechanism": (
            "Symmetric ciphers encrypt fast with a shared secret; public-key "
            "systems exchange keys and prove identities without a shared "
            "secret. Authenticated encryption binds confidentiality and "
            "integrity in one operation so neither can be left out by "
            "accident."
        ),
        "tradeoff": (
            "Stronger parameters resist longer-running attacks but cost "
            "more CPU and bandwidth. Post-quantum candidates use larger "
            "keys and signatures. Forward secrecy ties session keys to "
            "ephemeral exchanges at the cost of more handshake work."
        ),
        "practice": (
            "Practitioners use vetted libraries, prefer high-level APIs to "
            "low-level primitives, and rotate keys on a schedule. Custom "
            "constructions are reserved for cases where the standard "
            "library cannot express the requirement and reviewed externally."
        ),
    },
    {
        "title": "Hash Functions and Content Addressing",
        "background": (
            "A cryptographic hash function maps arbitrary-length input to "
            "a fixed-length digest, with strong collision resistance and "
            "pre-image resistance. Content-addressed systems use digests "
            "as primary identifiers for the data."
        ),
        "mechanism": (
            "Hashes underlie integrity checks, deduplication, build-system "
            "caches, version control, and certificate transparency. Merkle "
            "trees aggregate many hashes into one root so a small witness "
            "proves membership without revealing every leaf."
        ),
        "tradeoff": (
            "Stronger hashes are slower; truncated digests collide more "
            "often. Older deployments still rely on hashes that recent "
            "cryptanalysis has weakened, so migrations and dual-hashing "
            "transitions are recurring engineering work."
        ),
        "practice": (
            "Build systems and supply-chain tooling treat hashes as the "
            "fundamental name for an artifact. Reproducible builds make "
            "those names stable across machines, which is what lets remote "
            "caches share work safely."
        ),
    },
]


INSTRUCTION_HEADER = (
    "<｜begin▁of▁sentence｜>You are a careful and exhaustive "
    "technical writer. Stay in long-form analytical mode for the entire "
    "response; do not summarize, do not truncate, do not stop early.<｜User"
    "｜>"
    "Please write an extremely long and detailed multi-chapter analytical "
    "essay covering every topic listed below. For each topic, write at "
    "least eight paragraphs that go well beyond the background notes -- "
    "expand each claim into its historical context, the engineering "
    "tradeoffs, the implementation challenges, and a worked illustrative "
    "example. Use formal section headings of the form '## Chapter N: <Title>'. "
    "Do not summarize. Do not produce a table of contents. Begin with "
    "Chapter 1 immediately and continue through every topic in order. "
    "Aim for at least eight thousand words of essay output. Continue "
    "writing until you have addressed every topic; do not stop at the end "
    "of any one chapter.\n\n"
    "Topics and background notes follow.\n"
    "===================================\n"
)


INSTRUCTION_FOOTER = (
    "===================================\n"
    "Begin Chapter 1 now. Remember: at least eight paragraphs per chapter, "
    "no summaries, no early stops. Continue through all "
    "{topic_count} chapters.<｜Assistant｜>"
)


def render_topic(idx: int, t: dict[str, str]) -> str:
    return (
        f"\n## Topic {idx}: {t['title']}\n\n"
        f"Background. {t['background']}\n\n"
        f"Mechanism. {t['mechanism']}\n\n"
        f"Tradeoff. {t['tradeoff']}\n\n"
        f"Practice. {t['practice']}\n"
    )


def build_prompt() -> str:
    body_parts = [INSTRUCTION_HEADER]
    for i, topic in enumerate(TOPICS, start=1):
        body_parts.append(render_topic(i, topic))
    body_parts.append("\n")
    body_parts.append(INSTRUCTION_FOOTER.format(topic_count=len(TOPICS)))
    return "".join(body_parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "long_context_essay_prompt.txt",
        help="Where to write the prompt. Default: tests/long_context_essay_prompt.txt.",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Diff the generated content against --output and exit nonzero on drift.",
    )
    args = ap.parse_args(argv)

    content = build_prompt()
    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as e:
            print(f"build_essay_prompt --check: cannot read {args.output}: {e}", file=sys.stderr)
            return 2
        if existing != content:
            print(
                f"build_essay_prompt --check: {args.output} is out of date; "
                "re-run without --check to regenerate.",
                file=sys.stderr,
            )
            return 1
        print(f"build_essay_prompt --check: {args.output} is up to date")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(
        f"build_essay_prompt: wrote {args.output} "
        f"({len(content)} bytes, {len(TOPICS)} topics, "
        f"{content.count(chr(10))} lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
