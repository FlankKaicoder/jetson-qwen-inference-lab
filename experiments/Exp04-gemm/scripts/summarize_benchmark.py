import csv, statistics, sys
from collections import defaultdict
src, dst = sys.argv[1:3]
groups = defaultdict(list)
with open(src, newline='') as f:
    for r in csv.DictReader(f): groups[(r['M'], r['K'], r['N'])].append(r)
with open(dst, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['M','K','N','trials','mean_latency_ms','median_latency_ms','sample_std_latency_ms','cv_pct','min_latency_ms','max_latency_ms','mean_gflops','mean_actual_window_ms','total_launches'])
    for (m,k,n), rows in sorted(groups.items(), key=lambda x: tuple(map(int,x[0]))):
        lat = [float(r['latency_ms']) for r in rows]; g = [float(r['gflops']) for r in rows]; mean = statistics.mean(lat); std = statistics.stdev(lat) if len(lat)>1 else 0.0
        w.writerow([m,k,n,len(rows),f'{mean:.9f}',f'{statistics.median(lat):.9f}',f'{std:.9f}',f'{100*std/mean:.6f}',f'{min(lat):.9f}',f'{max(lat):.9f}',f'{statistics.mean(g):.6f}',f'{statistics.mean(float(r["actual_window_ms"]) for r in rows):.6f}',sum(int(r['measurement_iterations']) for r in rows)])
