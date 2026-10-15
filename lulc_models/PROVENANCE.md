# Plugin provenance

The plugin code under `lulc_models/<name>/` was adapted from research scripts that live outside
this repo. Verbatim copies of those scripts were tracked at commit `d18e5e7` under
`lulc_models/<name>/original/` and removed in the October 2026 cleanup; the hashes below identify
the exact versions the plugins were derived from. Nothing is read from these at run time.

## plotted_land — from `/mnt/cog-data/guj/pl/code/`

```
6ec52771aa15bd0fc36bcd5c72dc9e51adcb2a766827002a1c124b9138a155c5  pl_pipeline_guj.py
a31cb494a7d7893eb20d807b94529272cf96ff05d57e3b5743c47b174b2c4cc2  add_triplets.py
61e7e7622347170fd0607d0d2bceb5681d23528dcab30cd6f603b843bc3a944b  pl_deliver_guj.py
8fdf73f67a54a3bf3c780e78ba6100b258bfff260ce1d5c9061ce3de63c4853d  saltfilter_guj.py
adafaefebae491d7f9f5a5ccf97b5a3973e565e22668c08bf0fadfd9999e97d5  guj_pl.py
a5fde96aacb5b2144e98e04c18564975bcf1fd26ba4f033be27a42701cbc16b9  guj_pl_byregion.py
```

## substations — from `/mnt/drso-data/solar_lora/scripts/` (07_infer_corridor) and the Stage A/B analysis scripts

```
91ee24d60c3f4f5e4b87b6618c9a4d60799eeb153f67c3f326cadec3c3e0cb16  07_infer_corridor.py
6b2a648cf36b67ad25b638a7b66965110c3f2a132b65eb41cecc7697040ccf27  sA04_score.py
92261d36b76efe6e6ba7de7e1a5efb91c9c27ad2618972664a4523dacf487e13  sA06_layers.py
73e50d201b7e9c95c44bb551ab046f4560726c713afc0356bad86c020ffa382e  sA07_nisar.py
1a0405c09af6b83460047e2d9f57b071124ce990be14b2c9b4d057ce86482ebb  sA08_cascade.py
15760991ed9ad562848de3f325f7196f082062b52b09b0780ef19c346eb35124  sA09_cascade_real.py
946d2f20b7f17fbe95aa71df28add46ab916dd637687a8dbe3a269ca39599815  sA10_s1.py
3383e910631fd7aefb1603a29920034a6f382d6f3e9ee512139b55775ce7f804  sA11_final.py
1adc4d391a59091d617f4f2af3f6348fd328a6dc620e818b03c41c3ac9949909  stageB_config_run.yaml
2e42c6255f8241d665acd487dae117c8acfc7b269b73c78ba48482db60d618b9  stageB_hparams.yaml
```
