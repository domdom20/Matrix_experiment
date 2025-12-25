#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import pandas as pd
import logging
from pathlib import Path

SPARK_HOME = Path(os.environ.get("SPARK_HOME", "/usr/local/spark"))
SPARK_MASTER = "spark://172.23.166.133:7077"
PYTHON_PATH  = "/usr/bin/python3"
DRIVER_HOST = "172.23.166.133"

if SPARK_HOME.exists():
    os.environ.setdefault("SPARK_HOME", str(SPARK_HOME))
    os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH
    
    spark_python = SPARK_HOME / "python"
    py4j_zip = next((spark_python / "lib").glob("py4j-*-src.zip"), None)
    
    sys.path.insert(0, str(spark_python))
    if py4j_zip and py4j_zip.exists():
        sys.path.insert(0, str(py4j_zip))
else:
    sys.exit(1)

from pyspark.sql import SparkSession
from pyspark import SparkConf
from systemds.context import SystemDSContext

logging.getLogger("systemds").setLevel(logging.ERROR)

def clean_env():
    if 'SYSTEMDS_ROOT' in os.environ: del os.environ['SYSTEMDS_ROOT']
    if os.path.exists("SystemDS-config.xml"): os.remove("SystemDS-config.xml")

def get_spark(driver_mem):
    conf = SparkConf() \
        .setAppName("SystemDS_Op_Contrast") \
        .setMaster(SPARK_MASTER) \
        .set("spark.driver.host", DRIVER_HOST) \
        .set("spark.executor.memory", "4g") \
        .set("spark.executor.cores", "1") \
        .set("spark.driver.memory", driver_mem) \
        .set("spark.local.dir", "/tmp/spark")
    return SparkSession.builder.config(conf=conf).getOrCreate()

def run_experiment(mode_name, n, driver_mem, use_config=False):
    print(f"\nRunning: {mode_name}")
    print(f"    Driver Mem: {driver_mem} | Op: Element-wise (X * Y)")
    
    clean_env()
    
    if use_config:
        with open("SystemDS-config.xml", "w") as f:
            f.write("""<root>
            <sysds.execmode>spark</sysds.execmode>
            <sysds.cp.limit>10485760</sysds.cp.limit>
            </root>""")
    
    spark = get_spark(driver_mem)
    sds = SystemDSContext(spark)
    
    try:
        if not use_config:
             sds.rand(rows=1000, cols=1000).compute()

        start = time.time()
        
        X = sds.rand(rows=n, cols=n, sparsity=1.0, min=0, max=1)
        Y = sds.rand(rows=n, cols=n, sparsity=1.0, min=0, max=1)
        
        res = (X * Y).abs().sum().compute()
        
        end = time.time()
        duration = end - start
        print(f"    Done | Time: {duration:.4f} s")
        return duration
        
    finally:
        if hasattr(sds, 'close'): sds.close()
        spark.stop()
        clean_env()

def main():
    print(f"{'='*60}")
    print(f"SystemDS IO Bound Comparison (X * Y)")
    print(f"Cluster: {SPARK_MASTER}")
    print(f"{'='*60}")
    
    N = 10000 
    
    try:
        time_cp = run_experiment("Single Node CP (8G)", N, driver_mem="8g", use_config=False)
        
        time_spark = run_experiment("Dist Spark (2G+Limit)", N, driver_mem="2g", use_config=True)
        
        print("\n" + "="*60)
        print("Final Results (X * Y)")
        print("="*60)
        
        df = pd.DataFrame([
            {"Mode": "Single Node CP", "Time(s)": round(time_cp, 4), "Note": "No Network Overhead"},
            {"Mode": "Dist Spark", "Time(s)": round(time_spark, 4), "Note": "Network/SerDe Overhead"}
        ])
        
        print(df.to_string(index=False))
        print("-" * 60)
        
        if time_cp > 0:
            ratio = time_spark / time_cp
            print(f"Ratio: {ratio:.1f}x")
            if ratio > 5:
                print("Conclusion: Single node wins for IO bound tasks.")
            
    finally:
        clean_env()

if __name__ == "__main__":
    main()