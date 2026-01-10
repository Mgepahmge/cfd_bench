# CFD-Bench

#### INTRODUCTION
This is the source code of our submitted paper to VLDB 2026 "CFD-Bench: A CFD-driven Benchmark for Scientific Data Processing Using Database Approaches". It contains 8 workloads simulating 8 different CFD tasks on databases as a benchmark test.


#### HOW TO USE
1. Before running the benchmark, a user must download a package of CFD data from CFD Lifecycle Dataset (https://www.scidb.cn/en/detail?dataSetId=3553563d222d41998d7ccdd2ceff1bf9). A .zip package has three subfolders, the files in the "postprocessing" folder, i.e., the .dat files are the raw data set.

2. Execute the "LoadDataTo_DB.py" script to decode the .dat files and load them into different database systems. The principle of the loading process is as follows:

To run the data loading process, use the following command:

```bash
python -m src.demo.LoadDataTo_DB
```

2.1 A .dat file is a standard CFD post-processing data format, used in softwares such as Tecplot. The first objective of data loading is to decode the .dat file, extracting its geometry topology and cell values. This process is done using the "Dat_data_decoder.py" script.
2.2 Due to the inability of current DBMS on handling geometrical operations, the geometry info is stored and managed using a VTK file. The cell values are organized into multidimensional array, managed by various database systems.
2.3 For the details of data modelling, please refer to our paper, Section "5. Evaluation".

3. Execute workload.py to perform the benchmark test using the corresponding workload.

To run all workloads, use the following command:

```bash
python -m src.demo.workload
```


