"""Per-region deliver + saltfilter (same canonical scripts), then statewide merge.
The zone-level pl_deliver unions millions of footprints on one core; per region the unions stay local.
usage: guj_pl_byregion.py run|merge   (env NPAR)"""
import os, sys, subprocess, time, re
import pandas as pd, geopandas as gpd, pyogrio
from concurrent.futures import ProcessPoolExecutor
PL="/mnt/cog-data/guj/pl"; CODE=PL+"/code"; PY="/home/vivek_jain/miniforge3/envs/clay/bin/python"
ENV=dict(os.environ,PL_BUILDINGS="/home/vivek_jain/guj_buildings/guj_buildings.gpkg",PL_WORLDCOVER="/mnt/cog-data/guj/worldcover_2021.vrt",OMP_NUM_THREADS="1")
t0=time.time(); log=lambda *a: print("[%6.0fs]"%(time.time()-t0),*a,flush=True)
def todo():
    out=[]
    for rid in sorted(os.listdir(PL+"/regions")):
        d=PL+"/regions/"+rid; c=d+"/core.gpkg"
        if not os.path.exists(c) or os.path.exists(c+".empty") or os.path.exists(d+"/final.gpkg"): continue
        if pyogrio.read_info(c,layer="clusters")["features"]==0: continue
        out.append(rid)
    return out
def one(rid):
    d=PL+"/regions/"+rid
    with open(d+"/deliver_log.txt","w") as lf:
        subprocess.run(["nice","-n","5",PY,CODE+"/pl_deliver_guj.py",d+"/core.gpkg",d+"/deliver.gpkg"],stdout=lf,stderr=subprocess.STDOUT,env=ENV,check=True)
        subprocess.run([PY,CODE+"/saltfilter_guj.py",d+"/deliver.gpkg",d+"/final.gpkg.t.gpkg"],stdout=lf,stderr=subprocess.STDOUT,env=ENV,check=True)
    os.replace(d+"/final.gpkg.t.gpkg",d+"/final.gpkg"); return rid
if __name__=="__main__":
    if sys.argv[1]=="run":
        T=todo(); log("regions to deliver:",len(T))
        with ProcessPoolExecutor(int(os.environ.get("NPAR",5))) as ex:
            fs={ex.submit(one,r):r for r in T}
            for f in fs:
                try: log(f.result(),"ok")
                except Exception as e: log(fs[f],"FAILED",repr(e)[:200])
        log("DELIVER DONE")
    else:
        Cs=[];Ps=[];Ss=[]
        for rid in sorted(os.listdir(PL+"/regions")):
            f=PL+"/regions/%s/final.gpkg"%rid
            if not os.path.exists(f): continue
            C=gpd.read_file(f,layer="plotted_land_clusters"); P=gpd.read_file(f,layer="plotted_land_plots")
            if "removed_saltworks_clusters" in pyogrio.list_layers(f)[:,0]:
                S=gpd.read_file(f,layer="removed_saltworks_clusters"); S["region"]=rid; Ss.append(S.to_crs(4326))
            z=C.crs.to_epsg(); C["region"]=rid; P["region"]=rid; C["utm_epsg"]=z; P["utm_epsg"]=z
            C["_k"]=rid+":"+C.cluster_id; P["_k"]=rid+":"+P.cluster_id
            Cs.append(C.to_crs(4326)); Ps.append(P.to_crs(4326))
        C=pd.concat(Cs,ignore_index=True); P=pd.concat(Ps,ignore_index=True)
        rp=C.geometry.representative_point(); C=C.assign(_x=rp.x.round(3),_y=-rp.y.round(3)).sort_values(["_x","_y"]).reset_index(drop=True)
        C["new"]=["GJ-PLC-%05d"%(i+1) for i in range(len(C))]; mp=dict(zip(C._k,C.new))
        P["cluster_id"]=P._k.map(mp); P["plot_id"]=P.cluster_id+P.plot_id.str.extract(r"^PLC-\d+(-\d+)$")[0]
        C["cluster_id"]=C.new
        C=gpd.GeoDataFrame(C.drop(columns=["_k","_x","_y","new"]),crs=4326); P=gpd.GeoDataFrame(P.drop(columns=["_k"]),crs=4326)
        assert C.cluster_id.is_unique and P.plot_id.is_unique and P.cluster_id.notna().all()
        out=PL+"/Gujarat_Plotted_Land_minus_Kachchh_byregion.gpkg"
        if os.path.exists(out): os.remove(out)
        C.to_file(out,layer="plotted_land_clusters",driver="GPKG"); P.to_file(out,layer="plotted_land_plots",driver="GPKG")
        if Ss: pd.concat(Ss,ignore_index=True).pipe(gpd.GeoDataFrame,crs=4326).to_file(out,layer="removed_saltworks_clusters",driver="GPKG")
        log("merged clusters %d plots %d salt-removed %d -> %s"%(len(C),len(P),sum(len(s) for s in Ss),out))
        print(C.built_class.value_counts().to_string()); print("area ha by class:"); print(C.groupby("built_class").area_ha.sum().round(0).to_string())
