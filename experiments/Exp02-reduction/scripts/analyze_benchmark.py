#!/usr/bin/env python3
import csv, statistics, sys
from collections import defaultdict

def read_rows(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))

def mean(values):
    return statistics.fmean(values)

def write_summary(source, target):
    data = read_rows(source)
    groups = defaultdict(list)
    for row in data:
        groups[(row['version'], row['N'], row['block_size'])].append(row)
    fields = ['version','N','block_size','rounds','mean_latency_ms','median_latency_ms','sample_std_latency_ms','cv_pct','min_latency_ms','max_latency_ms','pass_count','total_pass_count','total_kernel_launch_count','first_stage_grid']
    with open(target, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (version,n,block), members in sorted(groups.items()):
            values = [float(item['mean_latency_ms']) for item in members]
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            writer.writerow({'version':version,'N':n,'block_size':block,'rounds':len(values),'mean_latency_ms':mean(values),'median_latency_ms':statistics.median(values),'sample_std_latency_ms':sd,'cv_pct':100*sd/mean(values),'min_latency_ms':min(values),'max_latency_ms':max(values),'pass_count':sum(item['correctness']=='PASS' for item in members),'total_pass_count':sum(int(item['total_pass_count']) for item in members),'total_kernel_launch_count':sum(int(item['total_kernel_launch_count']) for item in members),'first_stage_grid':members[0]['first_stage_grid']})

def choose(source, target):
    data = read_rows(source)
    groups = defaultdict(list)
    for row in data:
        if row['correctness'] == 'PASS':
            groups[(row['version'], row['block_size'])].append(float(row['mean_latency_ms']))
    by_version = defaultdict(list)
    for (version, block), values in groups.items(): by_version[version].append((mean(values), block))
    with open(target, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['version','primary_block','secondary_block','reason'])
        for version, ranked in sorted(by_version.items()):
            ranked.sort(); writer.writerow([version, ranked[0][1], ranked[1][1] if len(ranked)>1 else '', 'lowest three-round mean; secondary retained for tie review'])

if __name__ == '__main__':
    if len(sys.argv) != 4: raise SystemExit('usage: analyze_benchmark.py summary|choose INPUT OUTPUT')
    {'summary': write_summary, 'choose': choose}.get(sys.argv[1], lambda *_: (_ for _ in ()).throw(SystemExit('unknown mode')))(sys.argv[2], sys.argv[3])