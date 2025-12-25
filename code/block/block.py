#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import time as timer

# 环境配置
SPARK_HOME = Path(os.environ.get("SPARK_HOME", "/usr/local/spark"))
SPARK_MASTER = "spark://172.23.166.133:7077"
SPARK_UI_PORT = 4040
PYTHON_PATH = "/usr/bin/python3"
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

class BlockOnlyExperiment:
    def __init__(self):
        self.results_dir = "block_results"
        os.makedirs(self.results_dir, exist_ok=True)

        # 实验配置列表：矩阵大小、稀疏度、块大小
        self.experiments = [
            {"n": 1000, "density": 0.1,  "block_size": 500},
            {"n": 1000, "density": 0.01, "block_size": 500},
            {"n": 1000, "density": 1.0,  "block_size": 500},
            {"n": 3000, "density": 0.1,  "block_size": 1000},
            {"n": 3000, "density": 0.01, "block_size": 1000},
            {"n": 3000, "density": 1.0,  "block_size": 1000},
            {"n": 5000, "density": 0.1,  "block_size": 1000},
            {"n": 5000, "density": 0.01, "block_size": 1000},
            {"n": 5000, "density": 1.0,  "block_size": 1000},
        ]

    def get_resource_config(self):
        return {
            "executor_memory": "2g",
            "executor_cores": "1",
            "driver_memory": "2g",
            "partitions": 12
        }

    def get_spark_stage_count(self, app_id, spark_master_url=f"http://{DRIVER_HOST}:{SPARK_UI_PORT}", retries=5, delay=2):
        stages_url = f"{spark_master_url}/api/v1/applications/{app_id}/stages"
        for attempt in range(retries):
            try:
                response = requests.get(stages_url, timeout=10)
                if response.status_code == 200:
                    stages_data = response.json()
                    if stages_data:
                        return len([s for s in stages_data if s.get('status') == 'COMPLETE'])
                    return 0
            except:
                pass
            if attempt < retries - 1:
                timer.sleep(delay)
        return 3  # 默认估算

    def generate_blocks(self, sc, N, density, block_size, partitions):
        num_blocks = (N + block_size - 1) // block_size
        meta_data = [(r, c) for r in range(num_blocks) for c in range(num_blocks)]
        rdd_meta = sc.parallelize(meta_data, partitions)

        def create_numpy_block(index):
            r, c = index
            rows = min(block_size, N - r*block_size)
            cols = min(block_size, N - c*block_size)
            if density >= 1.0:
                mat = np.random.rand(rows, cols)
            else:
                mat = np.zeros((rows, cols))
                nnz = int(rows*cols*density)
                if nnz>0:
                    idx = np.random.choice(rows*cols, nnz, replace=False)
                    mat.flat[idx] = np.random.rand(nnz)
            return (r, c), mat

        return rdd_meta.map(create_numpy_block)

    def block_matrix_multiply(self, rdd_A, rdd_B, block_size, n):
        def prepare_A(pair):
            (r, c), block = pair
            return (c, (r, 'A', block))
        def prepare_B(pair):
            (r, c), block = pair
            return (r, (c, 'B', block))
        joined_rdd = rdd_A.map(prepare_A).join(rdd_B.map(prepare_B))

        def multiply_blocks(pair):
            c, ((r_a, _, block_a), (c_b, _, block_b)) = pair
            return ((r_a, c_b), np.dot(block_a, block_b))

        return joined_rdd.map(multiply_blocks).reduceByKey(lambda a, b: a+b)

    def run_experiment(self):
        all_results = []
        res_conf = self.get_resource_config()

        for i, exp in enumerate(self.experiments, 1):
            n, density, block_size = exp['n'], exp['density'], exp['block_size']
            exp_id = f"BlockOnly_Exp{i}_{n}x{n}_D{int(density*100)}"

            conf = SparkConf().setAppName(f"BlockOnly_{exp_id}").setMaster(SPARK_MASTER)\
                .set("spark.driver.host", DRIVER_HOST)\
                .set("spark.executor.memory", res_conf['executor_memory'])\
                .set("spark.executor.cores", res_conf['executor_cores'])\
                .set("spark.driver.memory", res_conf['driver_memory'])\
                .set("spark.default.parallelism", str(res_conf['partitions']))\
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
                .set("spark.kryoserializer.buffer.max", "512m")\
                .set("spark.local.dir", "/tmp/spark")\
                .set("spark.ui.showConsoleProgress", "false")
            spark = SparkSession.builder.config(conf=conf).getOrCreate()
            sc = spark.sparkContext
            sc.setLogLevel("ERROR")
            app_id = sc.applicationId

            try:
                start_time = time.time()
                start_cpu = psutil.cpu_percent()
                start_mem = psutil.virtual_memory().percent

                rdd_A = self.generate_blocks(sc, n, density, block_size, res_conf['partitions'])
                rdd_B = self.generate_blocks(sc, n, density, block_size, res_conf['partitions'])
                rdd_A.cache().count()
                rdd_B.cache().count()

                calc_start = time.time()
                result_rdd = self.block_matrix_multiply(rdd_A, rdd_B, block_size, n)
                count = result_rdd.count()
                calc_end = time.time()
                wall_time = calc_end - calc_start

                stage_count = self.get_spark_stage_count(app_id)
                end_cpu = psutil.cpu_percent()
                end_mem = psutil.virtual_memory().percent

                num_blocks = (n + block_size - 1) // block_size
                block_elements = block_size*block_size
                block_size_mb = (block_elements*8*density)/(1024*1024)
                shuffle_read_mb = num_blocks*block_size_mb*2
                shuffle_write_mb = num_blocks*block_size_mb*2

                all_results.append({
                    "Matrix Size": n,
                    "Density": density,
                    "Wall Time (s)": round(wall_time,2),
                    "Shuffle Read (MB)": round(shuffle_read_mb,2),
                    "Shuffle Write (MB)": round(shuffle_write_mb,2),
                    "Stage Count": stage_count,
                    "CPU (%)": round((start_cpu+end_cpu)/2,1),
                    "Memory (%)": round((start_mem+end_mem)/2,1),
                    "Result Blocks": count,
                    "Status": "Success"
                })

                rdd_A.unpersist()
                rdd_B.unpersist()
            except:
                all_results.append({
                    "Matrix Size": n,
                    "Density": density,
                    "Wall Time (s)": 0,
                    "Shuffle Read (MB)": 0.0,
                    "Shuffle Write (MB)": 0.0,
                    "Stage Count": 0,
                    "CPU (%)": 0,
                    "Memory (%)": 0,
                    "Result Blocks": 0,
                    "Status": "Failed"
                })
            finally:
                spark.stop()

        self.save_and_plot(all_results)

    def save_and_plot(self, results):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "block_only_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        df_plot = df[df['Status'] == 'Success']
        if df_plot.empty:
            print("⚠️ 没有成功实验可绘图")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14,10))

        matrix_sizes = sorted(df_plot['Matrix Size'].unique())
        densities = sorted(df_plot['Density'].unique())

        bar_width = 0.25
        x = np.arange(len(matrix_sizes))
        for i, d in enumerate(densities):
            subset = df_plot[df_plot['Density'] == d].sort_values('Matrix Size')
            values = [subset[subset['Matrix Size']==size]['Wall Time (s)'].iloc[0] for size in matrix_sizes]
            axes[0,0].bar(x + i*bar_width, values, width=bar_width, label=f'Density {d}')
        axes[0,0].set_xlabel('Matrix Size')
        axes[0,0].set_ylabel('Time (s)')
        axes[0,0].set_title('Block-Only: Wall Time')
        axes[0,0].set_xticks(x + bar_width, matrix_sizes)
        axes[0,0].legend()
        axes[0,0].grid(axis='y', alpha=0.3)

        stage_counts = df_plot['Stage Count'].unique()
        axes[0,1].bar(range(len(stage_counts)), [len(df_plot[df_plot['Stage Count']==sc]) for sc in stage_counts])
        axes[0,1].set_xlabel('Stage Count')
        axes[0,1].set_ylabel('Experiment Count')
        axes[0,1].set_title('Stage Count Distribution')
        axes[0,1].set_xticks(range(len(stage_counts)), stage_counts)
        axes[0,1].grid(axis='y', alpha=0.3)

        for d in densities:
            subset = df_plot[df_plot['Density']==d].sort_values('Matrix Size')
            axes[1,0].plot(subset['Matrix Size'], subset['Shuffle Read (MB)'], 'o-', label=f'Density {d}', linewidth=2)
        axes[1,0].set_xlabel('Matrix Size')
        axes[1,0].set_ylabel('Shuffle Read (MB)')
        axes[1,0].set_title('Block-Only: Shuffle Read vs Matrix Size')
        axes[1,0].set_yscale('log')
        axes[1,0].legend()
        axes[1,0].grid(alpha=0.3)

        for d in densities:
            subset = df_plot[df_plot['Density']==d].sort_values('Matrix Size')
            axes[1,1].plot(subset['Matrix Size'], subset['Wall Time (s)'], 's-', label=f'Density {d}', linewidth=2)
        axes[1,1].set_xlabel('Matrix Size')
        axes[1,1].set_ylabel('Wall Time (s)')
        axes[1,1].set_title('Block-Only: Wall Time vs Matrix Size')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)

        plt.tight_layout()
        img_path = os.path.join(self.results_dir, "block_only_performance.png")
        plt.savefig(img_path)
        print(f"📊 图表已生成: {img_path}")

if __name__=="__main__":
    exp = BlockOnlyExperiment()
    exp.run_experiment()

