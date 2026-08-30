#!/usr/bin/env python3
import csv, math, statistics, sys
from collections import defaultdict
if len(sys.argv) != 3: raise SystemExit('usage: analyze_pairs.py stability.csv output.csv')
with open(sys.argv[1], newline='') as f: rows=list(csv.DictReader(f))
groups=defaultdict(dict)
for row in rows:
    groups[(row['version'], row['block_size'], row['N'], row['round'])]=float(row['mean_latency_ms'])
# Compare versions using their frozen candidate blocks and common round IDs.
pairs=[('V1','V2'),('V1','V3'),('V1','V4'),('V1','V5'),('V1','V6'),('V1','V7'),('V2','V3'),('V3','V4'),('V4','V5'),('V5','V6'),('V6','V7')]
with open(sys.argv[2],'w',newline='') as f:
    writer=csv.writer(f); writer.writerow(['version_a','version_b','rounds','mean_delta_ms','sample_std_delta_ms','ci95_low_ms','ci95_high_ms','interpretation'])
    for a,b in pairs:
        av={r:v for (ver,block,n,r),v in groups.items() if ver==a}
        bv={r:v for (ver,block,n,r),v in groups.items() if ver==b}
        keys=sorted(set(av)&set(bv)); deltas=[av[k]-bv[k] for k in keys]
        if not deltas: continue
        md=statistics.fmean(deltas); sd=statistics.stdev(deltas) if len(deltas)>1 else 0.0; half=2.7764451051977987*sd/math.sqrt(len(deltas)); lo,hi=md-half,md+half
        interp='significant' if lo>0 or hi<0 else 'statistically indistinguishable'
        writer.writerow([a,b,len(deltas),md,sd,lo,hi,interp])