from pathlib import Path
import json,re,time,math,hashlib,gc,sys
from collections import defaultdict
from array import array
import numpy as np,osmium,osmnx as ox
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]) if len(sys.argv)>1 else root/'output/reviews/4.09-shortest-path/netherlands-pbf-20260924'
p=Path(sys.argv[2]) if len(sys.argv)>2 else out/'netherlands-260923.osm.pbf'
filt=ox._overpass._get_network_filter('walk')
exclusions=[(k,re.compile(v)) for k,v in re.findall(r'\["([^"]+)"!~"([^"]+)"\]',filt)]
assert len(exclusions)>=8
def accept(w):
 return w.tags.get('highway') is not None and not any(rx.search(w.tags.get(k,'')) for k,rx in exclusions)
counts={}
class Count(osmium.SimpleHandler):
 def __init__(self):super().__init__();self.ways=0
 def way(self,w):
  if not accept(w):return
  self.ways+=1
  for n in w.nodes:counts[n.ref]=min(2,counts.get(n.ref,0)+1)
  if self.ways%100000==0:print(f'Pass 1: {self.ways:,} ways; {len(counts):,} referenced nodes',flush=True)
start=time.perf_counter();c=Count();c.apply_file(str(p));raw_node_count=len(counts);print(f'Pass 1 complete: {c.ways:,} ways',flush=True)
coordinates={};us=array('q');vs=array('q');weights=array('d')
class Build(osmium.SimpleHandler):
 def __init__(self):super().__init__();self.ways=0;self.invalid=0;self.zero=0
 def way(self,w):
  if not accept(w):return
  self.ways+=1
  refs=list(w.nodes)
  if len(refs)<2:return
  if any(not n.location.valid() for n in refs):self.invalid+=1;return
  prev=refs[0];origin=prev.ref;acc=0.;coordinates[origin]=(prev.lon,prev.lat)
  ring_mid=len(refs)//2 if refs[0].ref==refs[-1].ref else -1
  for i,n in enumerate(refs[1:],1):
   lat1=math.radians(prev.lat);lat2=math.radians(n.lat)
   a=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(math.radians(n.lon-prev.lon)/2)**2
   acc+=2*6371009*math.asin(min(1,math.sqrt(a)))
   if counts[n.ref]>1 or i==len(refs)-1 or i==ring_mid:
    coordinates[n.ref]=(n.lon,n.lat)
    if origin!=n.ref:
     us.append(min(origin,n.ref));vs.append(max(origin,n.ref));weights.append(acc)
     if acc==0:self.zero+=1
    origin=n.ref;acc=0.
   prev=n
  if self.ways%100000==0:print(f'Pass 2: {self.ways:,} ways; {len(us):,} segments',flush=True)
b=Build();b.apply_file(str(p),locations=True,idx='flex_mem')
assert b.invalid==0,f'{b.invalid} accepted ways have missing coordinates'
del counts;gc.collect()
u=np.asarray(us,dtype=np.int64);v=np.asarray(vs,dtype=np.int64);length=np.asarray(weights)
order=np.lexsort((length,v,u));u=u[order];v=v[order];length=length[order]
keep=np.r_[True,(u[1:]!=u[:-1])|(v[1:]!=v[:-1])];u=u[keep];v=v[keep];length=length[keep]
ids=np.unique(np.r_[u,v]);xy=np.array([coordinates[int(i)] for i in ids]);del coordinates
ui=np.searchsorted(ids,u);vi=np.searchsorted(ids,v)
adj=coo_matrix((np.ones(len(ui)*2,dtype=np.int8),(np.r_[ui,vi],np.r_[vi,ui])),shape=(len(ids),len(ids))).tocsr()
nc,labels=connected_components(adj,directed=False)
sizes=np.bincount(labels);largest=np.argmax(sizes)
eligible=np.flatnonzero(labels==largest)
selected=np.random.default_rng(2025).choice(eligible,min(1000,len(eligible)),replace=False)
np.savez(out/'nl-walk.npz',osm_ids=ids,x=xy[:,0],y=xy[:,1],u=ui,v=vi,length_m=length,component=labels,selected=selected)
meta={'pbf_sha256':json.loads((out/'source.json').read_text())['sha256'],'filter':filt,'mode':'walk; bidirectional, as in OSMnx walking-network construction','retained_ways':c.ways,'raw_referenced_nodes':len(ids) if False else None,'nodes':len(ids),'undirected_edges':len(ui),'directed_arcs':2*len(ui),'components':int(nc),'largest_component_nodes':int(sizes[largest]),'largest_component_fraction':float(sizes[largest]/len(ids)),'largest_component_sizes':sorted(map(int,sizes),reverse=True)[:20],'selected_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),'elapsed_build_s':time.perf_counter()-start,'missing_coordinates':b.invalid,'zero_length_segments':b.zero,'simplification':'Retain way endpoints and shared nodes; sum great-circle segment lengths within each way; retain a midpoint in closed rings; minimum-length parallel edges. This is not identical to OSMnx cross-way simplification. Ferries and turn-restriction relations are not included.','sampling':'1000 nodes, seed 2025, largest connected component; entire national graph retained in benchmark'}
meta['raw_referenced_nodes']=raw_node_count
(out/'graph.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
