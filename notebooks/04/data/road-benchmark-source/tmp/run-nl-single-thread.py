from pathlib import Path
import subprocess, sys, json, statistics, platform
from datetime import datetime, timezone
import psutil
from importlib.metadata import version
root=Path(__file__).resolve().parents[1]
out=root/'output/reviews/4.09-shortest-path/netherlands-pbf-20260924'
meta={'started_utc':datetime.now(timezone.utc).isoformat(), 'environment_file':'environment.json',
 'python':platform.python_version(), 'pandana':version('pandana'),
 'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),
 'ram_GiB':psutil.virtual_memory().total/2**30,
 'protocol':'Three fresh processes, one warm-up and one measured matrix each. One OpenMP thread for construction and queries.',
 'concurrent_work':'Process inspection during the first run found only this Pandana benchmark and its launcher; the earlier NetworkX job had stopped at 75 rows. CPU affinity and background load were not controlled.',
 'timing':'Query includes pair arrays, public batch call, NumPy conversion and cleanup. Setup separate; disk load, warm-up and validation excluded.'}
(out/'single-thread-protocol.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
for block in range(1,4):
 print(f'Single-thread fresh process {block}/3',flush=True)
 subprocess.run([sys.executable,'-X','utf8',str(root/'tmp/nl-thread-worker.py'),'measure',str(out),'1',str(block)],check=True)
runs=[json.loads((out/f'block-{b}-threads-1.json').read_text()) for b in range(1,4)]
times=[r['records'][-1]['wall_s'] for r in runs]
setup=[r['preprocessing_s'] for r in runs]
result={**meta,'finished_utc':datetime.now(timezone.utc).isoformat(),
 'query_times_s':times,'median_query_s':statistics.median(times),'min_query_s':min(times),'max_query_s':max(times),
 'preprocessing_times_s':setup,'median_preprocessing_s':statistics.median(setup),
 'median_setup_plus_query_s':statistics.median([a+b for a,b in zip(times,setup)]),
 'max_error_m':max(row['max_error_m'] for r in runs for row in r['records'])}
(out/'single-thread-summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (out/'report.md').open('a',encoding='utf-8') as f:
 f.write('\n## Additional single-thread Pandana measurement\n\n'+meta['protocol']+' '+meta['concurrent_work']+' '+meta['timing']+'\n\n')
 f.write(f"Query median {result['median_query_s']:.3f} s (range {min(times):.3f}–{max(times):.3f} s); preprocessing median {statistics.median(setup):.3f} s. Median paired setup plus query {result['median_setup_plus_query_s']:.3f} s. All six matrices matched the reference with maximum error {result['max_error_m']:g} m. This uses hierarchies built with one thread; it is not the one-thread point on the earlier shared 24-thread-built hierarchy. See single-thread-summary.json and block-*-threads-1.json.\n")
print(json.dumps(result,indent=2),flush=True)
