#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SPARK_HOME = Path(os.environ.get("SPARK_HOME", "/usr/local/spark"))
SPARK_MASTER = "spark://172.23.166.133:7077"
DRIVER_HOST = "172.23.166.133"
PYTHON_PATH = "/usr/bin/python3"

sys.path.insert(0, os.path.join(SPARK_HOME, "python"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python", "lib", "py4j-0.10.9.7-src.zip"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python", "lib", "pyspark.zip"))

from pyspark.sql import SparkSession
from pyspark import SparkConf
import logging

class SparkBroadcastMatrixExperiment:
    def __init__(self):
        self.results_dir = "spark_broadcast_results"
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
            .setAppName("Spark_Broadcast_Matrix_Exp") \
            .setMaster(SPARK_MASTER) \
            .set("spark.driver.host", DRIVER_HOST) \
            .set("spark.executor.instances", "3") \
            .set("spark.executor.cores", "2") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.memory", "4g") \
            .set("spark.local.dir", "/data/spark/tmp") \
            .set("spark.shuffle.service.enabled", "true") \
            .set("spark.dynamicAllocation.enabled", "false") \
            .set("spark.python.executable", PYTHON_PATH)

        return SparkSession.builder.config(conf=conf).getOrCreate()

    def run_experiment(self):
        all_results = []

        print(f"{'='*80}")
        print(f"🚀 Spark 矩阵乘法实验（Broadcast优化）")
        print(f"�� 模式: Hybrid Execution")
        print(f"{'='*80}")

        spark = self.get_spark_session()
        sc = spark.sparkContext

        try:
            for i, exp in enumerate(self.experiments, 1):
                n = exp['n']
                density = exp['density']

                print(f"\n▶️  [{i}/9] 正在运行: {n}x{n} | 密度 {density}")

                try:
                    start_time = time.time()
                    start_cpu = psutil.cpu_percent()
                    start_mem = psutil.virtual_memory().percent

                    def generate_sparse_matrix(rows, cols, sparsity):
                        matrix = np.random.rand(rows, cols)
                        mask = np.random.rand(rows, cols) > sparsity
                        matrix[mask] = 0
                        return matrix

                    X = generate_sparse_matrix(n, n, density)
                    Y = generate_sparse_matrix(n, n, density)

                    Y_broadcast = sc.broadcast(Y)

                    def matrix_row_mult(row):
                        return row @ Y_broadcast.value

                    X_rdd = sc.parallelize(X, numSlices=spark.conf.get("spark.executor.instances"))
                    Z_rdd = X_rdd.map(matrix_row_mult)
                    Z = np.array(Z_rdd.collect())
                    res_val = Z.sum()

                    end_time = time.time()
                    wall_time = end_time - start_time

                    end_cpu = psutil.cpu_percent()
                    end_mem = psutil.virtual_memory().percent

                    print(f"    ✅ 耗时: {wall_time:.2f}s | 结果校验和: {res_val:.2f}")

                    all_results.append({
                        "Matrix Size": n,
                        "Density": density,
                        "Wall Time (s)": round(wall_time, 2),
                        "CPU (%)": round((start_cpu + end_cpu)/2, 1),
                        "Memory (%)": round((start_mem + end_mem)/2, 1),
                        "Status": "Success"
                    })

                    Y_broadcast.unpersist()

                except Exception as e:
                    print(f"    ❌ 实验失败: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "Matrix Size": n,
                        "Density": density,
                        "Wall Time (s)": 0,
                        "CPU (%)": 0,
                        "Memory (%)": 0,
                        "Status": "Failed"
                    })

                time.sleep(5)

        finally:
            spark.stop()
            self.save_and_plot(all_results)

    def save_and_plot(self, results):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "spark_broadcast_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 数据已保存: {csv_path}")
        print("\n📋 结果汇总:")
        print(df.to_string())

        df_plot = df[df['Status'] == 'Success']
        if len(df_plot) == 0: return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        matrix_sizes = sorted(df_plot['Matrix Size'].unique())
        densities = sorted(df_plot['Density'].unique())

        bar_width = 0.25
        x = np.arange(len(matrix_sizes))

        for i, d in enumerate(densities):
            subset = df_plot[df_plot['Density'] == d].sort_values('Matrix Size')
            values = []
            for size in matrix_sizes:
                size_data = subset[subset['Matrix Size'] == size]
                values.append(size_data['Wall Time (s)'].iloc[0] if len(size_data) > 0 else 0)

            axes[0].bar(x + i*bar_width, values, width=bar_width, label=f'Density {d}')

        axes[0].set_xlabel('Matrix Size')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Spark Broadcast: Matrix Mult Performance')
        axes[0].set_xticks(x + bar_width, matrix_sizes)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        for i, d in enumerate(densities):
            subset = df_plot[df_plot['Density'] == d].sort_values('Matrix Size')
            if len(subset) > 0:
                axes[1].plot(subset['Matrix Size'], subset['Wall Time (s)'],
                               's-', label=f'Density {d}', linewidth=2)

        axes[1].set_xlabel('Matrix Size')
        axes[1].set_ylabel('Wall Time (s)')
        axes[1].set_title('Spark Broadcast: Time vs Size')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "spark_broadcast_performance.png"))
        print(f"📊 图表已生成")

if __name__ == "__main__":
    exp = SparkBroadcastMatrixExperiment()
    exp.run_experiment()
