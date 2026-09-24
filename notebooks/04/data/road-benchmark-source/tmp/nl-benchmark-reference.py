from pathlib import Path
import os,time,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse import csr_matrix,save_npz,load_npz
from scipy.sparse.csgraph import dijkstra
root=Path(__file__).resolve().parents[1]
out=root/'output/reviews/4.09-shortest-path/netherlands-pbf-20260924'
def rows(indices):
 a=load_npz(out/'reference-graph.npz');selected=np.load(out/'selected.npy');answer=[]
 for i in indices:
  dist=dijkstra(a,directed=True,indices=int(selected[i]),return_predecessors=False)
  answer.append(dist[selected]/1000)
 return indices,np.asarray(answer)
if __name__=='__main__':
 data=np.load(out/'nl-walk.npz');u=data['u'];v=data['v'];length=(data['length_m']*1000).astype(np.int64)
 matrix=csr_matrix((np.r_[length,length],(np.r_[u,v],np.r_[v,u])),shape=(len(data['osm_ids']),)*2)
 save_npz(out/'reference-graph.npz',matrix);np.save(out/'selected.npy',data['selected'])
 ref=np.empty((len(data['selected']),len(data['selected'])));start=time.perf_counter()
 del matrix,data
 with ProcessPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(rows,list(range(i,min(i+25,len(ref))))) for i in range(0,len(ref),25)]
  done=0
  for f in as_completed(jobs):
   indices,values=f.result();ref[indices]=values;done+=len(indices)
   print(f'Independent SciPy Dijkstra reference: {done}/{len(ref)} rows',flush=True)
 assert np.isfinite(ref).all() and np.all(np.diag(ref)==0)
 np.save(out/'reference.npy',ref)
 (out/'reference.json').write_text(json.dumps({'method':'SciPy compiled Dijkstra on symmetric CSR, integer-millimetre costs; four independent validation workers','rows':len(ref),'elapsed_s':time.perf_counter()-start,'purpose':'Correctness reference, not a NetworkX performance measurement','max_distance_m':float(ref.max())},indent=2))
 print('Reference complete',flush=True)
