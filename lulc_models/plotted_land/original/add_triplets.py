"""Admit strict near-identical triplets as clusters: 3 unclustered rectangles, each rect>=0.88, edges <=3 m apart,
azimuth within 3 deg, area / long side / short side within +-15% (|log ratio|<=0.15), median plot <=10,000 m2.
usage: add_triplets.py IN_result.gpkg OUT_result.gpkg"""
import sys, os, numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely import STRtree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
IN,OUT=sys.argv[1],sys.argv[2]
R=gpd.read_file(IN,layer="rectangles"); C=gpd.read_file(IN,layer="clusters")
geom=R.geometry.values; mrr=shapely.oriented_envelope(geom); c=shapely.get_coordinates(mrr).reshape(-1,5,2)[:,:4]
e1=c[:,1]-c[:,0]; e2=c[:,2]-c[:,1]; l1=np.hypot(*e1.T); l2=np.hypot(*e2.T); L=np.maximum(l1,l2); S=np.minimum(l1,l2)
lv=np.where((l1>=l2)[:,None],e1,e2); az=np.degrees(np.arctan2(lv[:,1],lv[:,0]))%90
un=np.where(R.cluster.values<0)[0]; g=geom[un]
pl,pr=STRtree(g).query(g,predicate="dwithin",distance=3.0); m=pl<pr; pl,pr=pl[m],pr[m]
A=R.area_m2.values[un]; a_=az[un]; Lu=L[un]; Su=S[un]; rc=R.rect.values[un]
dz=np.abs(a_[pl]-a_[pr]); dz=np.minimum(dz,90-dz); q=lambda v:np.abs(np.log(v[pl]/v[pr]))<=0.15
ok=(dz<=3)&q(A)&q(Su)&q(Lu)&(rc[pl]>=0.88)&(rc[pr]>=0.88)
n,lab=connected_components(coo_matrix((np.ones(ok.sum()),(pl[ok],pr[ok])),shape=(len(g),len(g))),directed=False); cnt=np.bincount(lab)
nid=int(max(C.cluster.max(),R.cluster.max()))+1; rows=[]; geoms=[]
for k in np.where(cnt==3)[0]:
    ix=un[lab==k]
    if np.median(R.area_m2.values[ix])>10000: continue
    R.loc[ix,"cluster"]=nid
    cl=shapely.buffer(shapely.buffer(shapely.union_all(mrr[ix]),8.0,quad_segs=2),-8.0,quad_segs=2); a=R.area_m2.values[ix]
    rows.append(dict(cluster=nid,n_plots=3,plots_ha=a.sum()/1e4,foot_ha=cl.area/1e4,fill_ratio=a.sum()/cl.area)); geoms.append(cl); nid+=1
C2=pd.concat([C,gpd.GeoDataFrame(rows,geometry=geoms,crs=C.crs)],ignore_index=True)
if os.path.exists(OUT): os.remove(OUT)
R.to_file(OUT,layer="rectangles",driver="GPKG"); gpd.GeoDataFrame(C2,crs=C.crs).to_file(OUT,layer="clusters",driver="GPKG")
print("triplets admitted:",len(rows),"| clusters",len(C),"->",len(C2))
