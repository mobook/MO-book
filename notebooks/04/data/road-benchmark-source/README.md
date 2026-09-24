# Code behind the recorded Netherlands experiment

These are unchanged copies of the scripts used on Joaquim Gromicho's Windows machine on 24 September 2026. SHA-256 hashes are in `source-hashes.json`. The notebook contains the Amsterdam experiment itself; these scripts produced the separately recorded Netherlands results.

- [Pandana launcher](tmp/run-nl-single-thread.py): the loop calls blocking `subprocess.run` three times. Each run finishes before the next starts. Each child constructs its own hierarchy using one OpenMP thread, performs one warm-up matrix, then one measured matrix. The reported query time is the median of these three sequential repetitions.
- [Pandana worker](tmp/nl-thread-worker.py): sets `OMP_NUM_THREADS` before importing Pandana, verifies the runtime, times construction separately, and separates pair-array creation, the batch call, conversion and cleanup. Its other thread configurations and `shared` mode belong to earlier experiments and are not used by the single-thread launcher.
- [NetworkX worker](tmp/nl-networkx-background.py): performs one single-threaded run, processing origins sequentially and checkpointing outside the query timer. Although it supports resuming, the reported result has `resumed: false` and one session. The earlier `benchmark-nl-networkx.py` attempt stopped early and did not produce the reported result.
- [Windows background launcher](tmp/launch-nl-networkx-background.ps1): records how the completed NetworkX job was launched; its paths refer to the original machine.
- [PBF graph builder](tmp/build-nl-walk-pbf.py), [download script](tmp/download-nl-pbf.py), and [independent reference construction](tmp/nl-benchmark-reference.py) preserve the input and validation code. Validation work is outside the query timing.

The `output/reviews/4.09-shortest-path/netherlands-pbf-20260924/` subdirectory contains the original environment, input provenance and result metadata, including `single-thread-summary.json`, the three `block-*-threads-1.json` files, and `networkx-standalone/result.json`.

## Reusing the archived scripts

The original scripts infer a workspace root from their location under `tmp/`. This archive preserves that layout. From its root, the Pandana launcher is `python tmp/run-nl-single-thread.py`; the NetworkX worker is `python tmp/nl-networkx-background.py` (Windows-specific). Install the versions recorded in `environment.json` in a separate environment. The PowerShell launcher contains absolute paths and must be adapted if the archive is moved.

The large PBF, graph arrays and reference matrices are not duplicated in this source archive. The original machine retains them in the project's `output/reviews/4.09-shortest-path/netherlands-pbf-20260924/` directory. To reproduce the timings elsewhere, supply `nl-walk.npz`, `reference.npy` and `selected.npy` in the matching output directory and verify the recorded graph checksum. Preserve the original results before running: these scripts write results to that directory. To start a new NetworkX measurement, use a fresh `networkx-standalone` output directory rather than existing checkpoints.

This is the historical Windows experiment code; uploading the teaching notebook to Colab does not automatically execute this national benchmark.
