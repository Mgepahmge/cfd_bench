# CFD-Bench test scripts

All scripts derive the project root from their own location and export:

```bash
CFD_BENCH_DATA_ROOT=<project-root>/data
```

Run them from any working directory.

## 1. Ingest CFD data

```bash
./scripts/01_ingest_cfd.sh <CFD_DATA_PATH> <DATASET>
```

Runs CFD ingest for all four backends:

```text
postgresql iotdb tiledb vtk
```

## 2. Ingest H5 data

```bash
./scripts/02_ingest_h5.sh <H5_DATA_PATH> <DATASET>
```

Runs H5 ingest for all four backends.

## 3. Full W1-W11 test

```bash
./scripts/03_full_test.sh <DATASET> [DURATION=10] [OUTPUT_CSV]
```

Examples:

```bash
./scripts/03_full_test.sh Kvlcc_351K_Small
./scripts/03_full_test.sh Kvlcc_351K_Small 5
./scripts/03_full_test.sh Kvlcc_351K_Small 10 results.csv
```

If `OUTPUT_CSV` is omitted, `cfd-bench run` is invoked without `--output`.

## 4. Performance comparison

```bash
./scripts/04_performance_test.sh <DATASET>
```

Runs W2/W4/W5/W6/W8 for 10 seconds on all four backends. The benchmark program's normal stdout/stderr is hidden. While it runs, the script prints only the current workload, for example `[performance] Running W4 ...`. A temporary CSV is used internally and removed automatically.

For each matching `workload + operation + step`, the final table reports only:

```text
db_avg/vtk = mean(PostgreSQL, IoTDB, TileDB txns/s) / VTK txns/s
```

The identifying columns (`workload`, `operation`, `step`) are retained so each ratio can be matched to its benchmark case.
