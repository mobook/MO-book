# Road-network quick-profile records: 3 November 2026

These are the original records behind the saved Amsterdam quick-profile outputs
in `../../09-shortest-path-road-networks.ipynb`. They are not new measurements.
The session started its benchmark environment record at
`2026-11-03T12:05:47.540080+00:00`. Execution was local Windows Python, as recorded
in `manifest.json`; it was not a Colab run.

- `amsterdam-repetitions.csv`: 18 timing records, three repetitions for each
  method at sizes 10, 25 and 50; all recorded matrix errors are zero.
- `amsterdam-summary.csv`: median query times from those repetitions.
- `amsterdam-phases.csv`: Pandana phase medians from the same calls.
- `amsterdam-selected.npy`: selected graph-node IDs, in order.
- `manifest.json`: machine, packages, single-thread settings, input mode,
  checksums, acquisition dates and session/export timestamps.
- `netherlands-recorded-windows.csv`: the separately recorded September national
  results, copied by the export; this was not a new national measurement.

Recovered byte for byte from the original local session ZIP named
`session-2026-11-03T120547.540080_0000.zip`. The data inputs are already bundled
in `../road-network-recorded-results.zip`; neither graph nor geocoding bytes
are duplicated here. Graph SHA-256:
`21e5c176f32a90127110e8c9d0049271df5614620629cd5fb17f678a4eb3760b`.

The neighbouring input ZIP preserves the 24 September full-profile Amsterdam
run, with 42 timing records. Keep it separate from these 18 quick-profile rows.
Rerunning the notebook records new timings; it does not reproduce identical
wall times or rerun the national experiment. The original national PBF and
reference matrices are not bundled. The graph's known venue-coverage limitation
is unchanged and explained in the notebook.

Source data: OpenStreetMap contributors, ODbL 1.0.
Attribution and licence information: https://www.openstreetmap.org/copyright
The notebook explains graph coverage, direction, length units and timing scope.
