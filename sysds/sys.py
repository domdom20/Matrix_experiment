#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import logging
from pathlib import Path

# ==========================================
# 1. 环境配置 (与Baseline保持一致)
# ==========================================
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
    print(f"❌ Error: Spark not found at {SPARK_HOME}")
    sys.exit(1)

from pyspark.sql import SparkSession
from pyspark import SparkConf
from systemds.context import SystemDSContext

logging.getLogger("systemds").setLevel(logging.ERROR)

if 'SYSTEMDS_ROOT' in os.environ:
    del os.environ['SYSTEMDS_ROOT']

class SparkMetricCollector:
    def __init__(self, spark_context):
        self.base_url = f"{spark_context.uiWebUrl}/api/v1/applications/{spark_context.applicationId}"
    
    def get_metrics(self):
        try:
            stages_url = f"{self.base_url}/stages"
            resp = requests.get(stages_url, timeout=2)
            if resp.status_code != 200:
                return 0, 0, 0
            
            data = resp.json()
            completed_stages = [s for s in data if s['status'] == 'COMPLETE']
            stage_count = len(completed_stages)
            
            shuffle_read_bytes = 0
            shuffle_write_bytes = 0
            
            exec_url = f"{self.base_url}/allexecutors"
            resp_exec = requests.get(exec_url, timeout=2)
            if resp_exec.status_code == 200:
                exec_data = resp_exec.json()
                for executor in exec_data:
                    if executor['id'] != 'driver':
                        shuffle_read_bytes += executor.get('totalShuffleRead', 0)
                        shuffle_write_bytes += executor.get('totalShuffleWrite', 0)

            return stage_count, shuffle_read_bytes, shuffle_write_bytes
        except Exception:
            return 0, 0, 0

class SystemDSExperiment:
    def __init__(self):
        self.results_dir = "sysds_results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.experiments = [
            {"n": 1000, "density": 0.1},
            {"n": 1000, "density": 0.01},
            {"n": 1000, "density": 1.0},
            {"n": 3000, "density": 0.1},
            {"n": 3000, "density": 0.01},
            {"n": 3000, "density": 1.0},
            {"n": 5000, "density": 0.1},
            {"n": 5000, "density": 0.01},
            {"n": 5000, "density": 1.0},
        ]

    def get_spark_session(self):
        conf = SparkConf() \
            .setAppName("SystemDS_Matrix_Exp") \
            .setMaster(SPARK_MASTER) \
            .set("spark.driver.host", DRIVER_HOST) \
            .set("spark.executor.memory", "4g") \
            .set("spark.executor.cores", "2") \
            .set("spark.driver.memory", "4g") \
            .set("spark.local.dir", "/tmp/spark") \
            .set("spark.ui.showConsoleProgress", "false")
        return SparkSession.builder.config(conf=conf).getOrCreate()

    def run_experiment(self):
        all_results = []
        print(f"{'='*80}")
        print(f"Apache SystemDS Matrix Multiplication Experiment")
        print(f"Cluster: {SPARK_MASTER}")
        print(f"{'='*80}")

        # 显式创建配置好的SparkSession
        spark = self.get_spark_session()
        spark.sparkContext.setLogLevel("ERROR")
        
        monitor = SparkMetricCollector(spark.sparkContext)
        
        # 将SparkSession传递给SystemDSContext
        sds = SystemDSContext(spark)

        try:
            for i, exp in enumerate(self.experiments, 1):
                n = exp['n']
                density = exp['density']
                print(f"\n[{i}/9] Running: {n}x{n} | Density {density}")

                try:
                    # 预热/清理
                    time.sleep(1)
                    
                    start_time = time.time()
                    start_cpu = psutil.cpu_percent()
                    start_mem = psutil.virtual_memory().percent
                    s_stages, s_read, s_write = monitor.get_metrics()

                    # SystemDS 逻辑
                    X = sds.rand(rows=n, cols=n, sparsity=density, min=0, max=1)
                    Y = sds.rand(rows=n, cols=n, sparsity=density, min=0, max=1)
                    Z = X @ Y
                    res_val = Z.sum().compute()

                    end_time = time.time()
                    end_cpu = psutil.cpu_percent()
                    end_mem = psutil.virtual_memory().percent
                    e_stages, e_read, e_write = monitor.get_metrics()

                    wall_time = end_time - start_time
                    delta_stages = e_stages - s_stages
                    delta_read_mb = (e_read - s_read) / (1024 * 1024)
                    delta_write_mb = (e_write - s_write) / (1024 * 1024)

                    print(f"    Time: {wall_time:.2f}s | Stage: {delta_stages} | Shuffle: {delta_read_mb:.2f}MB")

                    all_results.append({
                        "Matrix Size": n,
                        "Density": density,
                        "Wall Time (s)": round(wall_time, 2),
                        "Shuffle Read (MB)": round(delta_read_mb, 2),
                        "Shuffle Write (MB)": round(delta_write_mb, 2),
                        "Stage Count": delta_stages,
                        "CPU (%)": round((start_cpu + end_cpu)/2, 1),
                        "Memory (%)": round((start_mem + end_mem)/2, 1),
                        "Status": "Success"
                    })

                except Exception as e:
                    print(f"    Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "Matrix Size": n,
                        "Density": density,
                        "Wall Time (s)": 0,
                        "Shuffle Read (MB)": 0,
                        "Shuffle Write (MB)": 0,
                        "Stage Count": 0,
                        "CPU (%)": 0,
                        "Memory (%)": 0,
                        "Status": "Failed"
                    })
                time.sleep(1)

        finally:
            try:
                if hasattr(sds, 'close'): sds.close()
            except: pass
            spark.stop()
            self.save_and_plot(all_results)

    def save_and_plot(self, results):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "systemds_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nData Saved: {csv_path}")
        print("\nResults:")
        print(df.to_string())
        
        df_success = df[df['Status'] == 'Success']
        if len(df_success) > 0:
            try:
                plt.figure(figsize=(10, 6))
                for d in sorted(df_success['Density'].unique()):
                    subset = df_success[df_success['Density'] == d].sort_values('Matrix Size')
                    if not subset.empty:
                        plt.plot(subset['Matrix Size'], subset['Wall Time (s)'], 'o-', label=f'Density {d}')
                plt.xlabel('Matrix Size')
                plt.ylabel('Time (s)')
                plt.title('SystemDS Performance')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.results_dir, "systemds_performance.png"))
                print("Chart Generated")
            except Exception as e:
                print(f"Plotting failed: {e}")

if __name__ == "__main__":
    exp = SystemDSExperiment()
    exp.run_experiment()