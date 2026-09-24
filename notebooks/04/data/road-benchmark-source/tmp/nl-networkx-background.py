"""One resumable full national NetworkX matrix; no AI or network calls."""
from pathlib import Path
import os, sys, json, time, traceback, hashlib, msvcrt
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'output/reviews/4.09-shortest-path/netherlands-pbf-20260924'
OUT = DATA/'networkx-standalone'
OUT.mkdir(exist_ok=True)
sys.stdout = (OUT/'worker.log').open('a', encoding='utf-8', buffering=1)
sys.stderr = sys.stdout
lock = (OUT/'worker.lock').open('a+b')
lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
try:
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    sys.exit('A benchmark is already running.')

def now(): return datetime.now(timezone.utc).isoformat()
def save(name, value):
    target=OUT/name; temp=target.with_suffix(target.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,target)

state={'status':'loading','pid':os.getpid(),'started_utc':now(),'completed_rows':0}
save('status.json',state)
try:
    os.environ['OMP_NUM_THREADS']='1'
    import numpy as np, networkx as nx, psutil, platform, gc
    from importlib.metadata import version
    proc=psutil.Process()
    environment=json.loads((DATA/'environment.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256((DATA/'nl-walk.npz').read_bytes()).hexdigest()
    assert digest==environment['graph_sha256']
    data=dict(np.load(DATA/'nl-walk.npz'))
    selected=data['selected']; reference=np.load(DATA/'reference.npy')
    assert np.array_equal(selected,np.load(DATA/'selected.npy'))
    assert hashlib.sha256(selected.tobytes()).hexdigest()==environment['selected_sha256']
    n=len(selected); assert reference.shape==(n,n) and n==1000
    start=time.perf_counter(); G=nx.DiGraph()
    G.add_nodes_from(range(len(data['osm_ids'])))
    for u,v,w in zip(data['u'],data['v'],data['length_m']):
        u=int(u);v=int(v);mm=int(w*1000)
        G.add_edge(u,v,length_mm=mm);G.add_edge(v,u,length_mm=mm)
    construction=time.perf_counter()-start
    assert G.number_of_edges()==2*len(data['u'])
    del data;gc.collect()
    session={'started_utc':state['started_utc'],'pid':os.getpid(),'python':platform.python_version(),
      'networkx':version('networkx'),'os':platform.platform(),'cpu':environment['cpu'],
      'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),
      'ram_GiB':psutil.virtual_memory().total/2**30,'graph_sha256':digest,
      'graph_construction_s':construction,'nodes':len(G),'arcs':G.number_of_edges(),
      'system_cpu_before_pct':psutil.cpu_percent(interval=.2),
      'conditions':'One single-threaded run, no warm-up. Background load and CPU affinity not controlled.',
      'timing':'Sum of row wall times: Dijkstra over full reachable component, extraction of 1000 destinations, row array construction, distance-dictionary cleanup. Graph construction, validation, checkpoint I/O, progress updates and final matrix assembly excluded. Resumed sessions disclosed.'}
    session_name=f'session-{os.getpid()}-{int(time.time())}.json'
    save(session_name,session)
    rowdir=OUT/'rows';rowdir.mkdir(exist_ok=True)
    timings=[];cpus=[];sessions=set();matrix=np.empty((n,n))
    for i,origin in enumerate(selected):
        target=rowdir/f'{i:04d}.npz'
        if target.exists():
            with np.load(target) as stored:
                assert int(stored['origin'])==int(origin)
                row=stored['distances'];wall=float(stored['wall_s']);cpu=float(stored['cpu_s'])
                row_session=str(stored['session'])
        else:
            cpu0=sum(proc.cpu_times()[:2]);start=time.perf_counter()
            lengths=nx.single_source_dijkstra_path_length(G,int(origin),weight='length_mm')
            row=np.array([lengths[int(d)]/1000 for d in selected])
            del lengths
            wall=time.perf_counter()-start;cpu=sum(proc.cpu_times()[:2])-cpu0
            row_session=session_name
            assert np.max(np.abs(row-reference[i]))<=1e-6
            temp=target.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez(stream,origin=int(origin),distances=row,wall_s=wall,cpu_s=cpu,session=row_session)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temp,target)
        assert row.shape==(n,) and np.max(np.abs(row-reference[i]))<=1e-6
        matrix[i]=row;timings.append(wall);cpus.append(cpu);sessions.add(row_session)
        state.update(status='running',completed_rows=i+1,query_s=sum(timings),updated_utc=now(),
          estimated_remaining_s=sum(timings)/(i+1)*(n-i-1),max_error_m=float(np.max(np.abs(matrix[:i+1]-reference[:i+1]))))
        save('status.json',state)
        if (i+1)%25==0 or i==0: print(f'{now()} {i+1}/{n} rows; query {sum(timings):.2f} s',flush=True)
    np.save(OUT/'matrix.npy',matrix)
    result={**session,'finished_utc':now(),'completed_rows':n,'query_s':sum(timings),
      'cpu_s':sum(cpus),'cpu_wall_ratio':sum(cpus)/sum(timings),'session_files':sorted(sessions),
      'resumed':len(sessions)>1,'max_error_m':float(np.max(np.abs(matrix-reference))),
      'memory_info':proc.memory_info()._asdict()}
    save('result.json',result)
    report=f"# NetworkX: full Netherlands graph\n\n{n} x {n} distances; {len(G):,} nodes, {G.number_of_edges():,} arcs.\n\nQuery: **{sum(timings):.3f} seconds ({sum(timings)/60:.2f} minutes)**.\n\n{session['timing']}\n\n{session['conditions']} Sessions: {len(sessions)}. Maximum reference error: {result['max_error_m']:g} m.\n\nEnvironment and construction times are saved in the session JSON files.\n"
    (OUT/'report.md').write_text(report,encoding='utf-8')
    state.update(status='completed',finished_utc=now());save('status.json',state)
    print(report,flush=True)
except BaseException:
    state.update(status='failed',updated_utc=now(),error=traceback.format_exc())
    save('status.json',state);traceback.print_exc();raise
