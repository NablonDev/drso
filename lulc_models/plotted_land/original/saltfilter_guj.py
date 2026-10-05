"""Remove salt-works clusters: 1 km ring (cluster buffer 1000 m minus 50 m) WorldCover water+wetland+mangrove+bare >= 0.60.
Built % is NOT used: salt heaps are detected as buildings by Google/MS.  usage: saltfilter.py IN.gpkg OUT.gpkg"""
import sys, os, numpy as np, pandas as pd, geopandas as gpd, rasterio, warnings
from rasterio.features import rasterize
from rasterio.windows import from_bounds
warnings.filterwarnings("ignore")
IN,OUT=sys.argv[1],sys.argv[2]; THR=0.60
WC=os.environ.get("PL_WORLDCOVER","/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N21E069_Map.tif")
C=gpd.read_file(IN,layer="plotted_land_clusters"); Pl=gpd.read_file(IN,layer="plotted_land_plots")
ring=gpd.GeoSeries(C.geometry.buffer(1000).difference(C.geometry.buffer(50)).values,crs=C.crs).to_crs(4326)
with rasterio.open(WC) as s:
    w=from_bounds(*ring.total_bounds,transform=s.transform); a=s.read(1,window=w); tr=s.window_transform(w)
sc=np.zeros(len(C))
for i,g in enumerate(ring.geometry):
    m=rasterize([(g,1)],out_shape=a.shape,transform=tr,fill=0,dtype="uint8").astype(bool); v=a[m]
    sc[i]=np.isin(v,(80,90,95,60)).mean() if len(v) else 0
C["ring_saltscape"]=sc.round(3)
salt=C.ring_saltscape>=THR
bad=set(C.cluster_id[salt])
if os.path.exists(OUT): os.remove(OUT)
keepC=C[~salt].drop(columns="ring_saltscape"); keepP=Pl[~Pl.cluster_id.isin(bad)]
keepC.to_file(OUT,layer="plotted_land_clusters",driver="GPKG"); keepP.to_file(OUT,layer="plotted_land_plots",driver="GPKG")
C[salt].to_file(OUT,layer="removed_saltworks_clusters",driver="GPKG")
print("%s: clusters %d -> %d (removed %d, %.0f ha) | plots %d -> %d (removed %d)"%(os.path.basename(OUT),len(C),len(keepC),salt.sum(),C[salt].area_ha.sum(),len(Pl),len(keepP),len(Pl)-len(keepP)))
print("  kept clusters: %.0f ha ; built_class counts:"%keepC.area_ha.sum(),keepP.built_class.value_counts().to_dict())
