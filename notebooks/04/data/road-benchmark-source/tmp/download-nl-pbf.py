from pathlib import Path
import urllib.request,hashlib,json,time
from datetime import datetime,timezone
r=Path(__file__).resolve().parents[1]
out=r/'output/reviews/4.09-shortest-path/netherlands-pbf-20260924';out.mkdir(exist_ok=True)
url='https://download.geofabrik.de/europe/netherlands-260923.osm.pbf'
p=out/'netherlands-260923.osm.pbf';part=p.with_suffix('.pbf.part')
expected=urllib.request.urlopen(url+'.md5',timeout=60).read().decode().split()[0]
if not p.exists():
 with urllib.request.urlopen(url,timeout=90) as response,part.open('wb') as f:
  total=int(response.headers.get('Content-Length',0));size=0;last=time.time()
  while chunk:=response.read(4*1024*1024):
   f.write(chunk);size+=len(chunk)
   if time.time()-last>20:print(f'Downloaded {size/1e6:.0f}/{total/1e6:.0f} MB',flush=True);last=time.time()
 part.rename(p)
md5=hashlib.md5();sha=hashlib.sha256()
with p.open('rb') as f:
 while chunk:=f.read(8*1024*1024):md5.update(chunk);sha.update(chunk)
assert md5.hexdigest()==expected
manifest={'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'bytes':p.stat().st_size,'md5':expected,'sha256':sha.hexdigest(),'licence':'OpenStreetMap contributors, ODbL 1.0','provider':'Geofabrik','source_data_through':'2026-09-23T20:22:04Z'}
(out/'source.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
