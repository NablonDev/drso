"""Plotted-land pipeline, clean path, one pass: roads -> snap joins (exact coords) -> blocks -> blue anchors
-> rectangle edge extensions -> [idealised OMBB lines] -> dead-end extensions -> comb end caps -> U-caps
-> rectangles (rule + context) -> H2 clustering (module rule).  Logic and parameters as in
plotted_land_detection_methodology.md §3-§14 and s55_moduleH.py.
usage: python pl_pipeline.py ROADS.gpkg PROB.vrt OUTDIR IDEAL(0/1)"""
import sys, os, time, numpy as np, pandas as pd, geopandas as gpd, shapely, rasterio, pyproj, networkx as nx
import shapely.ops
from shapely import STRtree
from shapely.ops import polygonize
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROADS, PROB, OUT, IDEAL = sys.argv[1], sys.argv[2], sys.argv[3].rstrip("/")+"/", int(sys.argv[4])
os.makedirs(OUT, exist_ok=True); CRS=int(os.environ.get("PL_CRS","32642")); t0=time.time()
def log(*a): print("[%6.1fs]"%(time.time()-t0),*a,flush=True)
os.environ.setdefault("GDAL_CACHEMAX","4096")
src=rasterio.open(PROB); to3857=pyproj.Transformer.from_crs(CRS,3857,always_xy=True)

def pmean(lines):
    """mean road probability along each straight 2-pt line, one sample per 0.6 m"""
    if not len(lines): return np.zeros(0)
    c=shapely.get_coordinates(lines).reshape(-1,2,2); L=np.hypot(*(c[:,1]-c[:,0]).T)
    n=np.maximum(4,(L/0.6).astype(int)); off=np.concatenate([[0],np.cumsum(n)])
    t=np.concatenate([np.linspace(0,1,k) for k in n]); idx=np.repeat(np.arange(len(n)),n)
    xy=c[idx,0]+(c[idx,1]-c[idx,0])*t[:,None]; X,Y=to3857.transform(xy[:,0],xy[:,1])
    v=np.fromiter((s[0] for s in src.sample(zip(X,Y))),dtype=float,count=len(X))/255.0
    return np.add.reduceat(v,off[:-1])/n

def faces_of(lines,minarea=200):
    segs=shapely.get_parts(shapely.union_all(lines))
    f=np.array(list(polygonize(segs)),dtype=object); a=shapely.area(f); k=a>=minarea; f=f[k]; a=a[k]
    o=shapely.oriented_envelope(f); rc=a/shapely.area(o)
    c=shapely.get_coordinates(o).reshape(-1,5,2); e1=c[:,1]-c[:,0]; e2=c[:,2]-c[:,1]; l1=np.hypot(*e1.T); l2=np.hypot(*e2.T)
    el=np.minimum(l1,l2)/np.maximum(l1,l2); lv=np.where((l1>=l2)[:,None],e1,e2); az=np.degrees(np.arctan2(lv[:,1],lv[:,0]))%90
    return segs,f,a,rc,el,az
def report(tag,rc,el,n): log("%-40s blocks %6d | rect>=0.85 %5d | rect>=0.80 %5d"%(tag,n,((rc>=.85)&(el>=.15)).sum(),((rc>=.80)&(el>=.15)).sum()))
def save(geoms,name,**cols): gpd.GeoDataFrame(cols,geometry=list(geoms),crs=CRS).to_file(OUT+"lines.gpkg",layer=name,driver="GPKG")

# ---------------- stage 1: snap joins from exact endpoints
g=gpd.read_file(ROADS,engine="pyogrio").to_crs(CRS)
log("roads %d edges, %.1f km"%(len(g),g.length.sum()/1000))
road=shapely.get_parts(shapely.union_all(g.geometry.values))
co=[shapely.get_coordinates(s) for s in road]
deg=Counter([tuple(c[0]) for c in co]+[tuple(c[-1]) for c in co])
d1=np.array([k for k,v in deg.items() if v==1]); log("dangles %d"%len(d1))
P=shapely.points(d1); tr=STRtree(road)
pi,si=tr.query(P,predicate="dwithin",distance=25.0)
dd=shapely.distance(P[pi],road[si]); m=dd>=0.05; pi,si,dd=pi[m],si[m],dd[m]
o=np.lexsort((dd,pi)); pi,si,dd=pi[o],si[o],dd[o]; first=np.r_[True,pi[1:]!=pi[:-1]]; pi,si,dd=pi[first],si[first],dd[first]
tgt=shapely.get_coordinates(shapely.ops.nearest_points(P[pi],road[si])[1])
J=shapely.linestrings(np.stack([d1[pi],tgt],1)); pj=pmean(J)
acc=(dd<=3.0)|(pj>=0.20); J=J[acc]
J=J[~pd.Series(shapely.to_wkb(shapely.normalize(J))).duplicated().values]
log("snap joins %d (%.1f km) of %d candidates"%(len(J),shapely.length(J).sum()/1000,len(acc))); save(J,"snap_joins")

# ---------------- stage 2: blocks, blue anchors
L=np.concatenate([g.geometry.values,J])
_,f,a,rc,el,az=faces_of(L); report("v4 blocks (roads + joins)",rc,el,len(f))
BLUE=f[(rc>=.85)&(el>=.15)]

# ---------------- stage 3: rectangle edge extensions (8 rays per anchor, 80 m, first hit, p>=0.2)
roads=L; rtree=STRtree(roads)
omb=shapely.oriented_envelope(BLUE); C=shapely.get_coordinates(omb).reshape(-1,5,2)[:,:4,:]
rays=[]
for i in range(4):
    p0=C[:,i,:]; p1=C[:,(i+1)%4,:]; v=p1-p0; Lv=np.maximum(np.hypot(*v.T),1e-9); u=v/Lv[:,None]
    rays.append(shapely.linestrings(np.stack([p1,p1+u*80],1))); rays.append(shapely.linestrings(np.stack([p0,p0-u*80],1)))
rays=np.concatenate(rays)
ql,qr=rtree.query(rays,predicate="intersects")
hit=shapely.intersection(rays[ql],roads[qr]); st=shapely.get_point(rays[ql],0); dist=shapely.distance(st,hit)
m=dist>=0.5; ql,hit,dist,st=ql[m],hit[m],dist[m],st[m]
o=np.lexsort((dist,ql)); ql,hit,dist,st=ql[o],hit[o],dist[o],st[o]; first=np.r_[True,ql[1:]!=ql[:-1]]; ql,hit,st=ql[first],hit[first],st[first]
e=shapely.ops.nearest_points(st,hit)[1]
X=shapely.linestrings(np.stack([shapely.get_coordinates(st),shapely.get_coordinates(e)],1)); X=X[shapely.length(X)>=0.5]
X=X[~pd.Series(shapely.to_wkb(shapely.normalize(X))).duplicated().values]; px=pmean(X); X=X[px>=0.20]
log("extensions: %d rays, %d accepted (%.1f km)"%(len(rays),len(X),shapely.length(X).sum()/1000)); save(X,"edge_extensions")
L=np.concatenate([L,X])
_,f,a,rc,el,az=faces_of(L); report("C  + edge extensions",rc,el,len(f))
if IDEAL:
    ombl=shapely.boundary(shapely.oriented_envelope(BLUE)); save(ombl,"idealised_lines"); L=np.concatenate([L,ombl])
    segs,f,a,rc,el,az=faces_of(L); report("E  + idealised outlines",rc,el,len(f))
else:
    segs=shapely.get_parts(shapely.union_all(L))

# ---------------- F1: straight dead ends extended to the first line (40 m, p>=0.2)
co=[shapely.get_coordinates(s) for s in segs]; cnt=Counter([tuple(c[0]) for c in co]+[tuple(c[-1]) for c in co])
R_=[];own=[]
for k,c in enumerate(co):
    ln=segs[k]
    if ln.length<15 or np.hypot(*(c[-1]-c[0]))/ln.length<0.90: continue
    for end,far in ((c[-1],ln.interpolate(ln.length-10)),(c[0],ln.interpolate(10))):
        if cnt[tuple(end)]!=1: continue
        v=end-np.array([far.x,far.y]); v/=np.hypot(*v); R_.append(shapely.linestrings([end,end+v*40])); own.append(k)
R_=np.array(R_,dtype=object); own=np.array(own); tree=STRtree(segs)
ql,qr=tree.query(R_,predicate="intersects"); m=qr!=own[ql]; ql,qr=ql[m],qr[m]
hit=shapely.intersection(R_[ql],segs[qr]); st=shapely.get_point(R_[ql],0); dist=shapely.distance(st,hit)
m=dist>=0.5; ql,hit,dist,st=ql[m],hit[m],dist[m],st[m]
o=np.lexsort((dist,ql)); ql,hit,st=ql[o],hit[o],st[o]; first=np.r_[True,ql[1:]!=ql[:-1]]; ql,hit,st=ql[first],hit[first],st[first]
e=shapely.ops.nearest_points(st,hit)[1]
DE=shapely.linestrings(np.stack([shapely.get_coordinates(st),shapely.get_coordinates(e)],1)); pd_=pmean(DE); DE=DE[pd_>=0.20]
log("dead-end extensions accepted %d of %d"%(len(DE),len(pd_))); save(DE,"deadend_extensions")
segsF,f,a,rc,el,az=faces_of(np.concatenate([L,DE])); report("F1 + dead-end extensions",rc,el,len(f))

# ---------------- G: comb end caps
co=[shapely.get_coordinates(s) for s in segsF]; cnt=Counter([tuple(c[0]) for c in co]+[tuple(c[-1]) for c in co])
Pp=[];U=[]
for k,c in enumerate(co):
    ln=segsF[k]
    if ln.length<30 or np.hypot(*(c[-1]-c[0]))/ln.length<0.90: continue
    for end,far in ((c[-1],c[0]),(c[0],c[-1])):          # heading = whole-lane chord (robust to end wiggles)
        if cnt[tuple(end)]!=1: continue
        v=end-far; Pp.append(end); U.append(v/np.hypot(*v))
Pp=np.array(Pp); U=np.array(U)
def mutual_pairs(a,b,lat):
    side=np.sign(U[a,0]*(Pp[b,1]-Pp[a,1])-U[a,1]*(Pp[b,0]-Pp[a,0])); best={}
    for i,j,d_,sd in zip(a,b,lat,side):
        for key,val in (((i,sd),(d_,j)),((j,-sd),(d_,i))):
            if key not in best or d_<best[key][0]: best[key]=val
    return sorted({(min(i,j),max(i,j)) for (i,sd),(d_,j) in best.items() if best.get((j,-sd),(0,-1))[1]==i})
pts=shapely.points(Pp); a,b=STRtree(pts).query(pts,predicate="dwithin",distance=170); m=a<b; a,b=a[m],b[m]
d=Pp[b]-Pp[a]; ax=(d*U[a]).sum(1); lat=np.abs(U[a,0]*d[:,1]-U[a,1]*d[:,0])
ok=((U[a]*U[b]).sum(1)>=np.cos(np.radians(10)))&(lat>=15)&(lat<=130)&(np.abs(ax)<=40); a,b,lat=a[ok],b[ok],lat[ok]
pairs=np.array(mutual_pairs(a,b,lat)).reshape(-1,2)
n_,lab=connected_components(coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(Pp),len(Pp))),directed=False); comb=np.bincount(lab)
caps=[];kinds=[]
for cid in np.where(comb>=3)[0]:
    idx=np.where(lab==cid)[0]; u=U[idx].mean(0); u/=np.hypot(*u); nrm=np.array([-u[1],u[0]]); s_=Pp[idx]@u; smax=s_.max(); ext=[]
    for k_,i in enumerate(idx):
        Lx=(smax-s_[k_])/max(U[i]@u,0.5); q=Pp[i]+U[i]*Lx; ext.append(q)
        if Lx>0.5: caps.append(shapely.linestrings([Pp[i],q])); kinds.append("lane_ext")
    ext=np.array(ext); ext=ext[np.argsort(ext@nrm)]; caps.append(shapely.linestrings(ext)); kinds.append("cap")
caps=np.array(caps,dtype=object); log("comb caps: %d combs"%(np.array(kinds)=="cap").sum())
if len(caps): save(caps,"endcaps",kind=kinds)

# ---------------- H: U-caps at the shorter arm
segs=shapely.get_parts(shapely.union_all(np.concatenate([segsF,caps]))) if len(caps) else segsF
co=[shapely.get_coordinates(s) for s in segs]; cnt=Counter([tuple(c[0]) for c in co]+[tuple(c[-1]) for c in co])
Gr=nx.Graph()
for k,c in enumerate(co):
    p_,q_=tuple(c[0]),tuple(c[-1])
    if p_!=q_ and (not Gr.has_edge(p_,q_) or Gr[p_][q_]["w"]>segs[k].length): Gr.add_edge(p_,q_,w=segs[k].length)
Pp=[];U=[];BASE=[];K=[]
for k,c in enumerate(co):
    ln=segs[k]
    if ln.length<30 or np.hypot(*(c[-1]-c[0]))/ln.length<0.90: continue
    for end,base,far in ((c[-1],c[0],c[0]),(c[0],c[-1],c[-1])):   # heading = whole-lane chord
        if cnt[tuple(end)]!=1 or cnt[tuple(base)]<2: continue
        v=end-far; Pp.append(end); U.append(v/np.hypot(*v)); BASE.append(tuple(base)); K.append(k)
Pp=np.array(Pp); U=np.array(U); K=np.array(K)
pts=shapely.points(Pp); a,b=STRtree(pts).query(pts,predicate="dwithin",distance=330); m=a<b; a,b=a[m],b[m]
d=Pp[b]-Pp[a]; ax=(d*U[a]).sum(1); lat=np.abs(U[a,0]*d[:,1]-U[a,1]*d[:,0])
ok=((U[a]*U[b]).sum(1)>=np.cos(np.radians(10)))&(lat>=15)&(lat<=130)&(np.abs(ax)<=200)&(K[a]!=K[b]); a,b,lat=a[ok],b[ok],lat[ok]
uc=[]
for i,j in mutual_pairs(a,b,lat):
    lim=1.6*np.hypot(*(Pp[j]-Pp[i]))+10
    try: dist=nx.shortest_path_length(Gr,BASE[i],BASE[j],weight="w") if BASE[i]!=BASE[j] else 0.0
    except nx.NetworkXNoPath: continue
    if dist>lim: continue
    u=U[i]+U[j]; u/=np.hypot(*u); sh,lo=(i,j) if Pp[i]@u<=Pp[j]@u else (j,i)
    q=shapely.ops.nearest_points(shapely.Point(Pp[sh]),segs[K[lo]])[1]; q=np.array([q.x,q.y]); cv=q-Pp[sh]; cl_=np.hypot(*cv)
    if cl_<15 or abs(np.degrees(np.arccos(abs(cv@u)/cl_))-90)>20: continue
    uc.append(shapely.linestrings([Pp[sh],q]))
uc=np.array(uc,dtype=object); log("U-caps %d"%len(uc))
if len(uc): save(uc,"ucaps")
segs2,f,ar,rc,el,az=faces_of(np.concatenate([segs,uc]) if len(uc) else segs); report("H  + U-caps",rc,el,len(f))

# ---------------- rectangles: rule + context
isr=(rc>=.80)&(el>=.15)
cand=np.where((rc>=0.65)&(rc<.80)&(el>=.15)&(ar<60000))[0]; anc=np.where(isr)[0]
cl,cr=STRtree(f[anc]).query(f[cand],predicate="dwithin",distance=2.0)
dz=np.abs(az[cand[cl]]-az[anc[cr]]); dz=np.minimum(dz,90-dz); lr=np.abs(np.log(ar[cand[cl]]/ar[anc[cr]]))
okc=np.unique(cand[cl[(dz<=15)&(lr<=0.5)]])
origin=np.where(isr,"rule","").astype(object); origin[okc]="context"
R=gpd.GeoDataFrame({"area_m2":ar,"rect":rc,"elong":el,"origin":origin},geometry=f,crs=CRS)[origin!=""]
R=R[R.area_m2<60000].reset_index(drop=True)
log("rectangles %d (rule %d + context %d)"%(len(R),(R.origin=="rule").sum(),(R.origin=="context").sum()))

# ---------------- H2 clustering (module rule)
def near_int(r): return np.min(np.abs(np.log(r[:,None]/np.array([1,2,3])[None,:])),1)<=0.20
geom=R.geometry.values; mrr=shapely.oriented_envelope(geom)
C=shapely.get_coordinates(mrr).reshape(-1,5,2)[:,:4,:]; e1=C[:,1]-C[:,0]; e2=C[:,2]-C[:,1]; l1=np.hypot(*e1.T); l2=np.hypot(*e2.T)
Lg=np.maximum(l1,l2); Sh=np.minimum(l1,l2); lv=np.where((l1>=l2)[:,None],e1,e2); ld=np.degrees(np.arctan2(lv[:,1],lv[:,0]))%180; az=ld%90
CX=shapely.get_coordinates(shapely.centroid(geom)); area=R.area_m2.values
pl,pr=STRtree(geom).query(geom,predicate="dwithin",distance=15.0); m=pl<pr; pl,pr=pl[m],pr[m]
gap=shapely.distance(geom[pl],geom[pr]); d=np.abs(az[pl]-az[pr]); dz=np.minimum(d,90-d); lr=np.abs(np.log(area[pl]/area[pr]))
v=CX[pr]-CX[pl]; bb=np.degrees(np.arctan2(v[:,1],v[:,0]))%180; r=np.abs(bb-ld[pl]); rel=np.minimum(r,180-r)
base=(gap<=15)&(dz<=15); keep=base&(~(rel>=55)|(lr<=0.5)); mod=np.zeros(len(pl),bool)
rr=lambda x,y: np.maximum(x,y)/np.minimum(x,y)
for A1,B1,A2,B2 in ((Lg[pl],Lg[pr],Sh[pl],Sh[pr]),(Lg[pl],Sh[pr],Sh[pl],Lg[pr])):
    r1=rr(A1,B1); r2=rr(A2,B2); mod|=((np.abs(np.log(r1))<=0.2)&near_int(r2))|((np.abs(np.log(r2))<=0.2)&near_int(r1))
mod&=base&~keep; k=keep|mod
n_,lab=connected_components(coo_matrix((np.ones(k.sum()),(pl[k],pr[k])),shape=(len(R),len(R))),directed=False); cc=np.bincount(lab)
rows=[];cg=[]
for cid in np.where(cc>=4)[0]:
    idx=np.where(lab==cid)[0]
    c_=shapely.buffer(shapely.buffer(shapely.union_all(mrr[idx]),8.0,quad_segs=2),-8.0,quad_segs=2); a_=area[idx]
    rows.append(dict(cluster=int(cid),n_plots=len(idx),plots_ha=a_.sum()/1e4,foot_ha=c_.area/1e4,fill_ratio=a_.sum()/c_.area)); cg.append(c_)
CL=gpd.GeoDataFrame(pd.DataFrame(rows),geometry=cg,crs=CRS)
R["cluster"]=np.where(np.isin(lab,CL.cluster.values),lab,-1)
log("clusters %d | plots clustered %d of %d | footprint %.0f ha | fill median %.3f p10 %.3f | module links %d"%(
    len(CL),(R.cluster>=0).sum(),len(R),CL.foot_ha.sum(),CL.fill_ratio.median(),CL.fill_ratio.quantile(.1),mod.sum()))
R.to_file(OUT+"result.gpkg",layer="rectangles",driver="GPKG"); CL.to_file(OUT+"result.gpkg",layer="clusters",driver="GPKG")
log("DONE")
