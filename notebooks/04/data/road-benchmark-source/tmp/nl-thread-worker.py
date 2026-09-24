import os,sys,json,time,gc,hashlib,platform,subprocess
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
mode=sys.argv[1]; out=Path(sys.argv[2]);threads=int(sys.argv[3]);block=int(sys.argv[4])
os.environ['OMP_NUM_THREADS']=str(threads)
os.environ['OMP_DYNAMIC']='FALSE'
import numpy as np,pandas as pd,networkx as nx,osmnx as ox,psutil
root=Path(__file__).resolve().parents[1]
graph=out/'nl-walk.npz'
data=dict(np.load(graph))  # Complete disk loading before the setup timer.
selected=data['selected']
import pandana
from threadpoolctl import threadpool_info
runtime=threadpool_info()
omp=[p for p in runtime if p.get('user_api')=='openmp']
assert omp and all(p['num_threads']==threads for p in omp),runtime
start=time.perf_counter()
nodes=pd.DataFrame({'x':data['x'],'y':data['y']})
construct=time.perf_counter()
network=pandana.Network(nodes.x,nodes.y,pd.Series(data['u']),pd.Series(data['v']),
                       pd.DataFrame({'length':data['length_m']}),twoway=True)
setup=time.perf_counter()-start;construction=time.perf_counter()-construct
reference=np.load(out/'reference.npy');assert np.array_equal(selected,np.load(out/'selected.npy'))
def query():
 start=time.perf_counter()
 origins=np.repeat(selected,len(selected));destinations=np.tile(selected,len(selected))
 t1=time.perf_counter()
 values=network.shortest_path_lengths(origins,destinations)
 t2=time.perf_counter()
 matrix=np.asarray(values).reshape(len(selected),-1)
 t3=time.perf_counter()
 del values,origins,destinations
 t4=time.perf_counter()
 return matrix,{'arrays_s':t1-start,'batch_call_s':t2-t1,'conversion_s':t3-t2,'cleanup_s':t4-t3}
records=[]
if mode=='shared':
 from threadpoolctl import threadpool_limits
 for b,order in enumerate([[8,16,24],[24,8,16],[16,24,8]],1):
  for t in order:
   with threadpool_limits(limits=t,user_api='openmp'):
    assert all(p['num_threads']==t for p in threadpool_info() if p['user_api']=='openmp')
    for rep in range(2):
     gc.collect();cpu=psutil.cpu_percent(interval=.1);proc=psutil.Process();cpu0=sum(proc.cpu_times()[:2]);start=time.perf_counter()
     result,phases=query();wall=time.perf_counter()-start;used=sum(proc.cpu_times()[:2])-cpu0
     error=float(np.max(np.abs(result-reference)));assert error<=1e-6
     record={'threads':t,'block':b,'warmup':rep==0,'wall_s':wall,'cpu_s':used,'cpu_wall_ratio':used/wall,'background_cpu_pct':cpu,'max_error_m':error,**phases}
     records.append(record);del result
    (out/'shared-hierarchy.json').write_text(json.dumps({'construction_threads':24,'preprocessing_s':setup,'records':records},indent=2))
    print('SHARED '+json.dumps(record),flush=True)
 sys.exit()
for repetition in range(2):
 gc.collect();cpu=psutil.cpu_percent(interval=.1);proc=psutil.Process();cpu0=sum(proc.cpu_times()[:2]);start=time.perf_counter()
 result,phases=query()
 wall=time.perf_counter()-start;used=sum(proc.cpu_times()[:2])-cpu0
 error=float(np.max(np.abs(result-reference)));assert error<=1e-6
 records.append({'threads':threads,'block':block,'warmup':repetition==0,'wall_s':wall,'cpu_s':used,'cpu_wall_ratio':used/wall,'background_cpu_pct':cpu,'max_error_m':error,**phases})
 del result
payload={'threads':threads,'block':block,'preprocessing_s':setup,'construction_s':construction,'openmp':omp,'records':records,'utc':datetime.now(timezone.utc).isoformat(),'memory_info':psutil.Process().memory_info()._asdict()}
(out/f'block-{block}-threads-{threads}.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(records[-1]),flush=True)
