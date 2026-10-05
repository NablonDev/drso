"""result.gpkg (rectangles, clusters) -> building-footprint coverage (Google v3 U Microsoft) -> client deliverable.
usage: python pl_deliver.py RESULT.gpkg OUT.gpkg"""
import sys, os, numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely import STRtree
W=os.path.dirname(os.path.abspath(__file__))+"/"
RES,OUT=sys.argv[1],sys.argv[2]
CL=gpd.read_file(RES,layer="clusters"); R=gpd.read_file(RES,layer="rectangles"); R=R[R.cluster>=0].reset_index(drop=True)
bb=tuple(gpd.GeoSeries([shapely.box(*CL.total_bounds)],crs=CL.crs).buffer(500).to_crs(4326).total_bounds)
FP=pd.concat([gpd.read_file(os.environ.get("PL_BUILDINGS",W+"gujarat_4layer_FINAL_corridor.gpkg"),layer=l,bbox=bb,engine="pyogrio").to_crs(CL.crs)[["geometry"]]
              for l in ("google_open_buildings_v3","microsoft_buildings_2026")],ignore_index=True)
FP=shapely.make_valid(FP.geometry.values); FP=FP[np.unique(STRtree(FP).query(CL.geometry.values,predicate="intersects")[1])]
U=shapely.get_parts(shapely.union_all(FP))
def cov(geoms):
    gi,pi=STRtree(U).query(geoms,predicate="intersects")
    a=shapely.area(shapely.intersection(geoms[gi],U[pi])); return np.minimum(np.bincount(gi,weights=a,minlength=len(geoms))/shapely.area(geoms),1.0)
R["built_pct"]=(100*cov(R.geometry.values)).round(1); CL["built_pct"]=(100*cov(CL.geometry.values)).round(1)
BINS=[-0.01,5,20,40,100.01]; LAB=["Vacant (<5% built)","Sparsely built (5-20%)","Partly built (20-40%)","Built-up (>=40%)"]
cat=lambda p: pd.cut(p,BINS,labels=LAB,right=False).astype(str)
c=CL.geometry.representative_point(); CL=CL.assign(_x=c.x.round(-2),_y=-c.y.round(-2)).sort_values(["_x","_y"]).reset_index(drop=True)
CL["cluster_id"]=["PLC-%04d"%(i+1) for i in range(len(CL))]; R["cluster_id"]=R.cluster.map(dict(zip(CL.cluster,CL.cluster_id)))
p=R.geometry.representative_point(); R=R.assign(_x=p.x,_y=-p.y).sort_values(["cluster_id","_x","_y"]).reset_index(drop=True)
R["plot_id"]=R.cluster_id+"-"+(R.groupby("cluster_id").cumcount()+1).map("{:03d}".format)
R["area_m2"]=R.geometry.area.round(0); R["built_class"]=cat(R.built_pct)
P=R[["plot_id","cluster_id","area_m2","built_pct","built_class","geometry"]]
v=P.groupby("cluster_id").agg(n_plots=("plot_id","size"),plots_area_ha=("area_m2",lambda a:a.sum()/1e4),
    vacant_plots=("built_class",lambda s:(s==LAB[0]).sum()),plot_built_pct_median=("built_pct","median"))
C=CL[["cluster_id","built_pct","geometry"]].merge(v,on="cluster_id")
C["area_ha"]=(C.geometry.area/1e4).round(2); C["plots_area_ha"]=C.plots_area_ha.round(2)
C["vacant_plots_pct"]=(100*C.vacant_plots/C.n_plots).round(1); C["built_class"]=cat(C.built_pct)
C=C[["cluster_id","n_plots","area_ha","plots_area_ha","built_pct","built_class","vacant_plots","vacant_plots_pct","plot_built_pct_median","geometry"]]
assert P.plot_id.is_unique and C.cluster_id.is_unique
if os.path.exists(OUT): os.remove(OUT)
gpd.GeoDataFrame(C,crs=CL.crs).to_file(OUT,layer="plotted_land_clusters",driver="GPKG")
gpd.GeoDataFrame(P,crs=CL.crs).to_file(OUT,layer="plotted_land_plots",driver="GPKG")
pd.set_option("display.width",200)
for nm,d,a in (("PLOTS",P,"area_m2"),("CLUSTERS",C,"area_ha")):
    s=d.groupby("built_class").agg(n=(d.columns[0],"size"),area=(a,"sum")).reindex(LAB); s["n_pct"]=(100*s.n/s.n.sum()).round(1); print(nm,len(d)); print(s.to_string())
print("clusters area %.0f ha ; wrote %s"%(C.area_ha.sum(),OUT))
