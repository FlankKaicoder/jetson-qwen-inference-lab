import csv, math, statistics, sys
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1], newline="")))
groups=defaultdict(list)
for r in rows: groups[(r["version"],int(r["width"]),int(r["height"]))].append(float(r["mean_ms"]))
print("version,width,height,mean_ms,median_ms,min_ms,max_ms,std_ms,cv_pct,p95_ms,effective_gb_s")
for (v,w,h), xs in sorted(groups.items(), key=lambda x:(x[0][1],x[0][2],x[0][0])):
 xs.sort(); mean=statistics.mean(xs); med=statistics.median(xs); p95=xs[min(len(xs)-1,math.ceil(.95*len(xs))-1)]; gb=2*w*h*4/(mean*1e6)
 print(f"{v},{w},{h},{mean:.9f},{med:.9f},{min(xs):.9f},{max(xs):.9f},{statistics.stdev(xs):.9f},{statistics.stdev(xs)/mean*100:.6f},{p95:.9f},{gb:.6f}")
