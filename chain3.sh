#!/bin/bash
cd /home/claude/exp/code
python3 -I run_exp2.py --e2 15 --lr2 1e-4 --seeds 9 --variants ce,cw,focal,os,hoc,hoa,ftc --tag main --save_models none >> ../out/ext_seeds5-9.log 2>&1
for a in mobilenetv3_large_100 efficientnet_b0; do python3 -I run_arch.py --arch $a --seeds 0,1,2 > ../out/arch_$a.log 2>&1; done
echo DONE > ../out/chain3.done
