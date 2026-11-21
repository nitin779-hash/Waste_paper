# Two-phase MobileNetV2 transfer learning on TrashNet: reproducibility package

Companion code for "Disentangling Fine-Tuning, Augmentation and Training Length in Two-Phase MobileNetV2 Transfer Learning: A Multi-Seed, Duplicate-Aware Study on TrashNet".

## Layout
- `code/` training, duplicate analysis, statistics and benchmark scripts
- `splits/` per-seed split files (`split_main_seed{0-9}.json` random, `split_grp_seed{0-4}.json` grouped) and `groups_cos080.json` (duplicate group id of each image, in the dataset's file order)
- `results/` per-seed metrics of every arm (`results_main.json`, `results_grp.json`, `results_arch_*.json`), `final_stats.json` (all statistics used in the paper), `contamination.json`, `edge_bench.json`
- `logs/` complete training logs
- `figures/` figure scripts and the two 300-dpi figures (`fig1_pipeline_300dpi.png`, `fig2_results_300dpi.png`), plus the table/paper builder

## Data
Download the TrashNet "dataset-resized" release (Thung & Yang, 2016), 2,527 images in six folders (cardboard, glass, metal, paper, plastic, trash). Edit `DATA` in `code/common.py` to point at it. Images are resized to 224x224 and normalised with ImageNet statistics.

## Weights
- MobileNetV2: ImageNet-1k weights of github.com/d-li14/mobilenetv2.pytorch (`mobilenetv2_1.0-0c6065bc.pth`), placed in `DATA/w/`.
- MobileNetV3-Large / EfficientNet-B0: timm release files `mobilenetv3_large_100_ra-f55367f5.pth`, `efficientnet_b0_ra-3dd342df.pth`, placed in `DATA/w2/`.

## Environment
`pip install -r requirements.txt` (PyTorch 2.15.0, timm 1.0.31, Python 3.13). All runs were CPU-only (2 vCPU).

## Reproduce
```
cd code
python3 run_exp.py ...                                  # seeds 0-4: arms p1_ce p1_cw p2_ce p2_cw p2_focal p2_os (original run)
python3 run_exp2.py --e2 15 --lr2 1e-4 --seeds 0,1,2,3,4,5,6,7,8,9 --variants ce,cw,focal,os,hoc,hoa,ftc --tag main --save_models none
python3 dup_analysis.py && python3 dup2.py              # embeddings, duplicate groups, contamination
python3 run_exp2.py --e2 15 --lr2 1e-4 --seeds 0,1,2,3,4 --variants ce,cw,os,ftc,hoa --tag grp --grouped ../results/groups_cos080.json
python3 run_arch.py --arch mobilenetv3_large_100 --seeds 0,1,2 ; python3 run_arch.py --arch efficientnet_b0 --seeds 0,1,2
python3 stats_final.py grp                              # writes out/final_stats.json
python3 bench_edge.py                                   # size, MACs, server-CPU latency
cd ../figures && python3 figs2.py && python3 build2.py  # figures and paper tables
```
Scripts read and write `/home/claude/exp/{data,out}`; change `OUT` and `DATA` in `code/common.py` and the paths at the top of each script if you use another location. See `chain2.sh`/`chain3.sh` for the exact command lines used. Seeds: 0-9 (random splits), 0-4 (grouped splits and contamination analysis).

## Notes
- Latency numbers are server-CPU measurements, not mobile or embedded.
- Per-image test probabilities of seeds 0-4 of the first run were deleted by mistake; metrics and confusion matrices are unaffected and the contamination analysis uses the FT-C probabilities (`testprobs_main_seed*_ftc.npy`).
- Repository URL: to be inserted by the author.
