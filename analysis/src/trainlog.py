"""Training curves as a CSV next to the checkpoint, so they can be plotted after
training without parsing SLURM logs. One row per --log_every iterations: the same
numbers the log prints, plus spread, timing and the SLURM job id.

    log = CsvLog(os.path.join(out, 'train_log.csv'), fields, keep_upto=start - 1)
    log.row(it=200, loss=0.0035, ...)

Resume-safe. A resumed job restarts from the last checkpoint (every --ckpt_every
iterations), so the iterations between that checkpoint and the kill run twice.
On (re)start, rows with it > keep_upto are dropped, so each iteration appears
once, from the run that continued. keep_upto=0 (a fresh run) starts a new file.
If the columns changed since the file was written, the old file is kept as
<name>.old and a new one started.

Every row is flushed immediately: a job killed by TIMEOUT or OOM keeps its curve.
Pure python, no torch.
"""
import csv, os


class CsvLog:
    def __init__(self, path, fields, keep_upto=0):
        self.path, self.fields = path, list(fields)
        rows, same = [], True
        if os.path.exists(path) and keep_upto > 0:
            with open(path, newline='') as f:
                r = csv.DictReader(f)
                same = r.fieldnames == self.fields
                if same:
                    rows = [row for row in r if row.get('it') and int(row['it']) <= keep_upto]
            if not same:
                os.replace(path, path + '.old')
        tmp = path + '.tmp'                      # rewrite atomically, then append
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader(); w.writerows(rows)
        os.replace(tmp, path)
        self.kept = len(rows)
        self.f = open(path, 'a', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=self.fields)

    def row(self, **kv):
        self.w.writerow({k: kv.get(k, '') for k in self.fields})
        self.f.flush()
