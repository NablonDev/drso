"""Statewide plotted land (Gujarat minus Kachchh) = canonical Bhuj-Lakadia v2 pipeline run region by region.
 regions : ~40 km EPSG:3857 squares + 3 km overlap; UTM 42N if centre lon < 72 E else 43N
 region  : roads (vec/*.geojson of blocks touching the region) -> pl_pipeline_guj.py (IDEAL=1, variant A)
           -> add_triplets.py -> keep clusters whose representative point is in the region core
 zone    : merge cores -> pl_deliver_guj.py (Google v3 + MS footprints) -> saltfilter_guj.py (WorldCover VRT)
 state   : merge zones -> EPSG:4326, statewide IDs GJ-PLC-nnnnn
usage: guj_pl.py regions|run|zones|merge   (env NPAR)"""
import os, sys, re, glob, json, subprocess, time
import numpy as np, pandas as pd, geopandas as gpd, shapely
from concurrent.futures import ProcessPoolExecutor
G="/mnt/cog-data/guj"; PL=G+"/pl"; CODE=PL+"/code"; PY="/home/vivek_jain/miniforge3/envs/clay/bin/python"
O=20037508.342789244; TM=2*O/2**18; RSZ=40000.0; OV=3000.0
PROB=G+"/prob_u_all.vrt"; BLD="/home/vivek_jain/guj_buildings/guj_buildings.gpkg"; WC=G+"/worldcover_2021.vrt"
t0=time.time(); log=lambda *a: print("[%6.0fs]"%(time.time()-t0),*a,flush=True)
def regions():
    z=np.load(G+"/guj_mask_z18.npz"); M=z["M"]; c0=int(z["c0"]); r0=int(z["r0"])
    x0=-O+c0*TM; y1=O-r0*TM; x1=x0+M.shape[1]*TM; y0=y1-M.shape[0]*TM
    out=[]
    for i,yy in enumerate(np.arange(y1,y0,-RSZ)):
        for j,xx in enumerate(np.arange(x0,x1,RSZ)):
            ca=int((xx-x0)//TM); cb=int(min((xx+RSZ-x0)//TM+1,M.shape[1])); ra=int((y1-yy)//TM); rb=int(min((y1-yy+RSZ)//TM+1,M.shape[0]))
            if M[ra:rb,ca:cb].any():
                lon=np.degrees((xx+RSZ/2)/6378137.0)
                out.append(dict(rid="r%02d_%02d"%(i,j),x0=xx,y0=yy-RSZ,x1=xx+RSZ,y1=yy,crs=32642 if lon<72 else 32643))
    return pd.DataFrame(out)
BL=re.compile(r"^(\d+)_(\d+)\.geojson$")
def blocks_in(x0,y0,x1,y1):
    out=[]
    for f in os.listdir(G+"/vec"):
        m=BL.match(f)
        if not m: continue
        bx,by=int(m.group(1)),int(m.group(2)); bx0=-O+bx*TM; by1=O-by*TM
        if bx0<x1 and bx0+16*TM>x0 and by1>y0 and by1-16*TM<y1: out.append(G+"/vec/"+f)
    return out
def one(r):
    d=PL+"/regions/"+r["rid"]; os.makedirs(d,exist_ok=True)
    if os.path.exists(d+"/core.gpkg"): return r["rid"],"cached"
    fs=blocks_in(r["x0"]-OV,r["y0"]-OV,r["x1"]+OV,r["y1"]+OV)
    if os.path.exists(d+"/out/result_trip.gpkg"):
        import pyogrio; return _core(r,d,pyogrio.read_info(d+"/roads.gpkg")["features"])
    feats=[]
    for f in fs: feats+=json.load(open(f)).get("features",[])
    if len(feats)<50: open(d+"/core.gpkg.empty","w").write("%d\n"%len(feats)); gpd.GeoDataFrame(geometry=[],crs=r["crs"]).to_file(d+"/core.gpkg",layer="clusters"); return r["rid"],"few roads"
    g=gpd.GeoDataFrame.from_features(feats,crs=3857)
    g=g.clip(shapely.box(r["x0"]-OV,r["y0"]-OV,r["x1"]+OV,r["y1"]+OV)); g=g[g.geom_type=="LineString"]
    g.to_file(d+"/roads.gpkg",layer="roads",driver="GPKG")
    # region-local VRT: the statewide one has 29k sources and point sampling through it crawls
    ul=[]
    for f in fs:
        t=os.path.basename(f)[:-8]; u=G+"/prob_u/u_%s.tif"%t
        if os.path.exists(u): ul.append(u)
    open(d+"/prob_list.txt","w").write("\n".join(ul)+"\n")
    subprocess.run(["gdalbuildvrt","-q","-input_file_list",d+"/prob_list.txt",d+"/prob.vrt"],check=True)
    env=dict(os.environ,PL_CRS=str(r["crs"]),GDAL_CACHEMAX="2048",OMP_NUM_THREADS="1")
    if not os.path.exists(d+"/out/result_trip.gpkg"):          # reuse a finished pipeline run
      with open(d+"/log.txt","w") as lf:
        subprocess.run(["nice","-n","10",PY,CODE+"/pl_pipeline_guj.py",d+"/roads.gpkg",d+"/prob.vrt",d+"/out","1"],stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
        subprocess.run([PY,CODE+"/add_triplets.py",d+"/out/result.gpkg",d+"/out/result_trip.gpkg"],stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    return _core(r,d,len(g))
def _core(r,d,nroads):
    R=gpd.read_file(d+"/out/result_trip.gpkg",layer="rectangles"); C=gpd.read_file(d+"/out/result_trip.gpkg",layer="clusters")
    rp=gpd.GeoSeries(C.geometry.representative_point(),crs=C.crs).to_crs(3857)
    core=(rp.x>=r["x0"])&(rp.x<r["x1"])&(rp.y>=r["y0"])&(rp.y<r["y1"])
    C=C[core.values].copy(); R=R[R.cluster.isin(C.cluster)].copy()
    off=int(r["rid"][1:3])*1000+int(r["rid"][4:6])            # region-unique cluster ids
    C["cluster"]=C.cluster.astype("int64")+off*1_000_000; R["cluster"]=R.cluster.astype("int64")+off*1_000_000; C["region"]=r["rid"]; R["region"]=r["rid"]
    R.to_file(d+"/core.gpkg.t",layer="rectangles",driver="GPKG"); C.to_file(d+"/core.gpkg.t",layer="clusters",driver="GPKG")
    os.replace(d+"/core.gpkg.t",d+"/core.gpkg")
    return r["rid"],"clusters %d rects %d roads %d"%(len(C),len(R),nroads)
if __name__=="__main__":
    step=sys.argv[1]; RG=regions()
    if step=="regions":
        print(len(RG),"regions"); print(RG.crs.value_counts().to_string())
        gpd.GeoDataFrame(RG,geometry=[shapely.box(a,b,c,d) for a,b,c,d in RG[["x0","y0","x1","y1"]].values],crs=3857).to_file(PL+"/regions.gpkg",driver="GPKG")
    elif step=="run":
        # progressive: a region starts once every block touching it (with overlap) is vectorised
        UB=re.compile(r"^u_(\d+)_(\d+)\.tif$"); done=set(); failed={}
        def ready(r):
            vs=set(os.listdir(G+"/vec"))
            for f in os.listdir(G+"/prob_u"):
                m=UB.match(f)
                if not m: continue
                bx,by=int(m.group(1)),int(m.group(2)); bx0=-O+bx*TM; by1=O-by*TM
                if bx0<r["x1"]+OV and bx0+16*TM>r["x0"]-OV and by1>r["y0"]-OV and by1-16*TM<r["y1"]+OV and "%d_%d.geojson"%(bx,by) not in vs: return False
            return True
        recs=RG.to_dict("records")
        while len(done)+len(failed)<len(recs):
            vec_busy=any(os.popen("pgrep -f guj_vec.py").read().split())
            npar=int(os.environ.get("NPAR_BUSY",2)) if vec_busy else int(os.environ.get("NPAR",8))
            todo=[r for r in recs if r["rid"] not in done and failed.get(r["rid"],0)<2 and ready(r)][:npar*3]
            if not todo: time.sleep(300); continue
            with ProcessPoolExecutor(npar) as ex:
                futs={ex.submit(one,r):r["rid"] for r in todo}
                for f in futs:
                    rid=futs[f]
                    try: _,msg=f.result(); done.add(rid); log(rid,msg,"| done %d/%d"%(len(done),len(recs)))
                    except Exception as e:
                        failed[rid]=failed.get(rid,0)+1; log(rid,"FAILED",repr(e)[:200])
                        if failed[rid]>=2: done.add(rid)
        log("REGIONS DONE failed:",[k for k,v in failed.items() if v>=2])
    elif step=="zones":
        for crs in sorted(RG.crs.unique()):
            ids=RG.rid[RG.crs==crs]; Rs=[];Cs=[]
            for rid in ids:
                p=PL+"/regions/%s/core.gpkg"%rid
                if not os.path.exists(p) or os.path.exists(p+".empty"): continue
                Cs.append(gpd.read_file(p,layer="clusters")); Rs.append(gpd.read_file(p,layer="rectangles"))
            zf=PL+"/zone_%d"%crs
            pd.concat(Rs,ignore_index=True).pipe(gpd.GeoDataFrame,crs=crs).to_file(zf+"_result.gpkg",layer="rectangles",driver="GPKG")
            pd.concat(Cs,ignore_index=True).pipe(gpd.GeoDataFrame,crs=crs).to_file(zf+"_result.gpkg",layer="clusters",driver="GPKG")
            env=dict(os.environ,PL_BUILDINGS=BLD,PL_WORLDCOVER=WC)
            subprocess.run([PY,CODE+"/pl_deliver_guj.py",zf+"_result.gpkg",zf+"_deliver.gpkg"],env=env,check=True)
            subprocess.run([PY,CODE+"/saltfilter_guj.py",zf+"_deliver.gpkg",zf+"_final.gpkg"],env=env,check=True)
        log("ZONES DONE")
    elif step=="merge":
        Cs=[];Ps=[]
        for crs in (32642,32643):
            f=PL+"/zone_%d_final.gpkg"%crs
            if not os.path.exists(f): continue
            C=gpd.read_file(f,layer="plotted_land_clusters"); P=gpd.read_file(f,layer="plotted_land_plots")
            C["utm_zone"]=crs; P["utm_zone"]=crs
            C["_k"]="%d:"%crs+C.cluster_id; P["_k"]="%d:"%crs+P.cluster_id
            Cs.append(C.to_crs(4326)); Ps.append(P.to_crs(4326))
        C=pd.concat(Cs,ignore_index=True); P=pd.concat(Ps,ignore_index=True)
        rp=C.geometry.representative_point(); C=C.assign(_x=rp.x.round(3),_y=-rp.y.round(3)).sort_values(["_x","_y"]).reset_index(drop=True)
        C["new"]=["GJ-PLC-%05d"%(i+1) for i in range(len(C))]; mp=dict(zip(C._k,C.new))
        P["cluster_id"]=P._k.map(mp); P["plot_id"]=P.cluster_id+P.plot_id.str.extract(r"^PLC-\d+(-\d+)$")[0]
        C["cluster_id"]=C.new
        C=gpd.GeoDataFrame(C.drop(columns=["_k","_x","_y","new"]),crs=4326); P=gpd.GeoDataFrame(P.drop(columns=["_k"]),crs=4326)
        assert C.cluster_id.is_unique and P.plot_id.is_unique
        out=PL+"/Gujarat_Plotted_Land_minus_Kachchh.gpkg"
        if os.path.exists(out): os.remove(out)
        C.to_file(out,layer="plotted_land_clusters",driver="GPKG"); P.to_file(out,layer="plotted_land_plots",driver="GPKG")
        log("merged clusters %d plots %d -> %s"%(len(C),len(P),out))
