#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import psutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import time as timer
from pathlib import Path

# ==========================================
# 1. 环境配置
# ==========================================
SPARK_HOME = Path(os.environ.get("SPARK_HOME", "/usr/local/spark"))
SPARK_MASTER = "spark://172.23.166.133:7077"  # 集群Master地址
SPARK_UI_PORT = 4040  # Spark UI端口
PYTHON_PATH  = "/usr/bin/python3"  # 统一Python 3.7路径
DRIVER_HOST = "172.23.166.133"

# 配置Spark环境
if SPARK_HOME.exists():
    os.environ.setdefault("SPARK_HOME", str(SPARK_HOME))
    os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH
    
    spark_python = SPARK_HOME / "python"
    py4j_zip = next((spark_python / "lib").glob("py4j-*-src.zip"), None)
    
    sys.path.insert(0, str(spark_python))
    if py4j_zip and py4j_zip.exists():
        sys.path.insert(0, str(py4j_zip))
    
    print(f"✅ Spark环境配置完成: {SPARK_HOME}")
else:
    print(f"❌ 错误: 未找到Spark目录 {SPARK_HOME}")
    sys.exit(1)

from pyspark.sql import SparkSession
from pyspark import SparkConf

class BaselineMatrixExperiment:
    def __init__(self):
        self.results_dir = "baseline_results"
        os.makedirs(self.results_dir, exist_ok=True)

        # 集群配置信息
        self.cluster_config = {
            "master": SPARK_MASTER,
            "driver_host": DRIVER_HOST,
            "python_path": PYTHON_PATH,
            "workers": 3,  
            "cores_per_worker": 1,  # 每个worker有1个core
            "memory_per_worker": "4.0 GiB",  # 每个worker有4GB内存
            "total_cores": 3,  # 总核心数 = 3 workers * 1 core
            "total_memory": "12.0 GiB"  # 总内存 = 3 workers * 4GB
        }

        # 7组实验配置
        self.experiments = [
            # 1000x1000
            {"n": 1000, "density": 0.1},
            {"n": 1000, "density": 0.01},
            {"n": 1000, "density": 1.0},

            # 3000x3000
            {"n": 3000, "density": 0.1},
            {"n": 3000, "density": 0.01},

            # 5000x5000
            {"n": 5000, "density": 0.1},
            {"n": 5000, "density": 0.01},
        ]

    def get_resource_config(self):

        return {
            "executor_memory": "2g",
            "executor_cores": "1",
            "driver_memory": "2g",
            "num_executors": "3",
            "partitions": 12,
        }


    def get_spark_stage_count(self, app_id, spark_master_url="http://172.23.166.133:4040", retries=5, delay=2):
        """
        从Spark Web UI获取当前实验的Stage数量
        """
        stages_url = f"{spark_master_url}/api/v1/applications/{app_id}/stages"

        for attempt in range(retries):
            try:
                response = requests.get(stages_url, timeout=10)
                if response.status_code == 200:
                    stages_data = response.json()

                    if stages_data:
                        completed_stages = [stage for stage in stages_data
                                          if stage.get('status') == 'COMPLETE']
                        return len(completed_stages)
                    else:
                        return 0
                else:
                    print(f"    ⚠️  获取Stage信息失败 (HTTP {response.status_code})，第{attempt+1}次重试...")
            except Exception as e:
                print(f"    ⚠️  连接Spark UI失败: {e}，第{attempt+1}次重试...")

            if attempt < retries - 1:
                timer.sleep(delay)

        print("    ⚠️  无法获取Stage数量，返回估算值")
        return 2  

    def generate_sparse_matrix(self, sc, N, density, partitions):
        """生成稀疏矩阵元素RDD"""
        print(f"    📊 生成 {N}x{N} 矩阵，密度 {density}")

        # 计算非零元素数量
        total_elements = N * N
        nnz = int(total_elements * density)

        if nnz == 0:
            return sc.parallelize([], partitions)

        # 生成非零元素
        indices = np.random.choice(total_elements, nnz, replace=False)
        rows = indices // N
        cols = indices % N
        values = np.random.rand(nnz)

        # 创建元素列表
        elements = list(zip(zip(rows, cols), values))

        return sc.parallelize(elements, partitions)

    def naive_matrix_multiply(self, rdd_A, rdd_B, N):

        # 准备数据：将A和B都转换为((i,j), value)格式
        # A: 矩阵A的元素
        # B: 矩阵B的元素

        # 执行笛卡尔积：将A和B的所有元素配对
        cartesian_rdd = rdd_A.cartesian(rdd_B)

        # 过滤和计算：只计算符合矩阵乘法条件的配对
        # 矩阵乘法条件：A的列索引 = B的行索引
        def filter_and_compute(pair):
            ((i, j), a_val), ((k, l), b_val) = pair
            if j == k:  # A的列索引等于B的行索引
                return (((i, l), a_val * b_val),)
            else:
                return ()

        # 执行计算
        intermediate_rdd = cartesian_rdd.flatMap(filter_and_compute)

        # 聚合相同位置的结果
        result_rdd = intermediate_rdd.reduceByKey(lambda a, b: a + b)

        return result_rdd

    def optimized_baseline_multiply(self, rdd_A, rdd_B, N):

        # 准备A：按列索引分组
        def prepare_A(pair):
            (i, j), val = pair
            return (j, (i, val))  # key为列索引

        # 准备B：按行索引分组
        def prepare_B(pair):
            (i, j), val = pair
            return (i, (j, val))  # key为行索引

        # 执行join操作
        joined_rdd = rdd_A.map(prepare_A).join(rdd_B.map(prepare_B))

        # 计算乘法结果
        def compute_product(pair):
            j, ((i, a_val), (k, b_val)) = pair
            # A[i][j] * B[j][k] = C[i][k]
            return ((i, k), a_val * b_val)

        result_rdd = joined_rdd.map(compute_product).reduceByKey(lambda a, b: a + b)      

        return result_rdd

    def run_experiment(self, use_optimized=True):
        all_results = []
        res_conf = self.get_resource_config()

        print(f"{'='*80}")
        print(f"🚀 Baseline矩阵乘法测试（无Block、无Broadcast）")
        print(f"🔧 使用{'优化版' if use_optimized else '朴素版'}算法")
        print(f"🔧 配置参数: ExecMem={res_conf['executor_memory']}, Partitions={res_conf['partitions']}")
        print(f"{'='*80}")

        try:
            for i, exp in enumerate(self.experiments, 1):
                n = exp['n']
                density = exp['density']
                exp_id = f"Baseline_Exp{i}_{n}x{n}_D{int(density*100)}"

                print(f"\n▶️  [{i}/{len(self.experiments)}] 正在运行: {n}x{n} | 密度 {density}")

                # 创建Spark Session
                conf = (SparkConf()
                    .setAppName(f"Baseline_{exp_id}")
                    .setMaster("spark://172.23.166.133:7077")
                    .set("spark.driver.host", "172.23.166.133")
                    .set("spark.executor.memory", res_conf['executor_memory'])
                    .set("spark.executor.cores", res_conf['executor_cores'])
                    .set("spark.driver.memory", res_conf['driver_memory'])
                    .set("spark.default.parallelism", str(res_conf['partitions']))        
                    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")                    
                    .set("spark.kryoserializer.buffer.max", "512m")
                    .set("spark.local.dir", "/tmp/spark")
                    .set("spark.ui.showConsoleProgress", "false"))

                spark = SparkSession.builder.config(conf=conf).getOrCreate()
                sc = spark.sparkContext
                sc.setLogLevel("ERROR")

                app_id = sc.applicationId
                print(f"    📱 Spark App ID: {app_id}")

                try:
                    start_time = time.time()
                    start_cpu = psutil.cpu_percent()
                    start_mem = psutil.virtual_memory().percent

                    # 计算矩阵大小
                    matrix_size_mb = (n * n * 8 * density) / (1024 * 1024)
                    print(f"    📊 矩阵估算大小: {matrix_size_mb:.2f}MB")

                    if matrix_size_mb > 100 and not use_optimized:
                        print(f"    ⚠️  警告：朴素算法对较大矩阵可能非常慢！")

                    # 1. 生成数据
                    gen_start = time.time()
                    print(f"    🌀 生成矩阵A和B...")
                    rdd_A = self.generate_sparse_matrix(sc, n, density, res_conf['partitions'])
                    rdd_B = self.generate_sparse_matrix(sc, n, density, res_conf['partitions'])

                    rdd_A.cache().count()
                    rdd_B.cache().count()

                    gen_time = time.time() - gen_start
                    print(f"    ⏱️  数据生成耗时: {gen_time:.2f}s")

                    calc_start = time.time()

                    # 2. 执行矩阵乘法
                    print(f"    🧮 开始矩阵乘法计算...")
                    if use_optimized:
                        result_rdd = self.optimized_baseline_multiply(rdd_A, rdd_B, n)    
                    else:
                        result_rdd = self.naive_matrix_multiply(rdd_A, rdd_B, n)

                    # 触发计算
                    print(f"    📈 计算最终结果...")
                    count = result_rdd.count()

                    calc_end = time.time()
                    compute_time = calc_end - calc_start
                    total_time = calc_end - start_time

                    # 3. 获取Stage Count指标
                    stage_count = self.get_spark_stage_count(app_id)

                    # 4. 记录指标
                    end_cpu = psutil.cpu_percent()
                    end_mem = psutil.virtual_memory().percent


                    nnz_A = n * n * density
                    nnz_B = n * n * density

                    if use_optimized:
                        # join操作：Shuffle两边的数据
                        shuffle_read_mb = (nnz_A + nnz_B) * 8 / (1024 * 1024)
                        shuffle_write_mb = (nnz_A * density + nnz_B * density) * 8 / (1024 * 1024)
                    else:
                        # Cartesian product：Shuffle所有数据
                        shuffle_read_mb = (nnz_A + nnz_B) * 8 / (1024 * 1024)
                        shuffle_write_mb = (nnz_A * nnz_B) * 8 / (1024 * 1024) * density  

                    print(f"\n    ✅ 实验完成!")
                    print(f"    ⏱️  总耗时: {total_time:.2f}s (计算: {compute_time:.2f}s)"
)
                    print(f"    📈 Stage数: {stage_count} | 结果元素数: {count}")
                    print(f"    💾 Shuffle估算: Read={shuffle_read_mb:.2f}MB, Write={shuffle_write_mb:.2f}MB")
                    print(f"    🖥️  CPU利用率: {round((start_cpu + end_cpu)/2, 1)}%")     

                    all_results.append({
                        "Matrix Size": n,
                        "Density": density,
                        "Total Time (s)": round(total_time, 2),
                        "Compute Time (s)": round(compute_time, 2),
                        "Gen Time (s)": round(gen_time, 2),
                        "Shuffle Read (MB)": round(shuffle_read_mb, 2),
                        "Shuffle Write (MB)": round(shuffle_write_mb, 2),
                        "Stage Count": stage_count,
                        "CPU (%)": round((start_cpu + end_cpu)/2, 1),
                        "Memory (%)": round((start_mem + end_mem)/2, 1),
                        "Result Elements": count,
                        "Algorithm": "Optimized-Baseline" if use_optimized else "Naive-Baseline",
                        "Status": "Success"
                    })

                    # 清理内存
                    rdd_A.unpersist()
                    rdd_B.unpersist()

                except Exception as e:
                    print(f"    ❌ 实验失败: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "Matrix Size": n,
                        "Density": density,
                        "Total Time (s)": 0,
                        "Compute Time (s)": 0,
                        "Gen Time (s)": 0,
                        "Shuffle Read (MB)": 0.0,
                        "Shuffle Write (MB)": 0.0,
                        "Stage Count": 0,
                        "CPU (%)": 0,
                        "Memory (%)": 0,
                        "Result Elements": 0,
                        "Algorithm": "Optimized-Baseline" if use_optimized else "Naive-Baseline",
                        "Status": f"Failed: {str(e)[:50]}..."
                    })
                finally:
                    spark.stop()
                    print(f"    🔄 清理Spark Session...")

                timer.sleep(2)

        except Exception as e:
            print(f"❌ 发生全局错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.save_and_plot(all_results, use_optimized)

    def save_and_plot(self, results, use_optimized):
        df = pd.DataFrame(results)
        algo_type = "optimized" if use_optimized else "naive"
        csv_path = os.path.join(self.results_dir, f"baseline_{algo_type}_metrics.csv")    
        df.to_csv(csv_path, index=False)
        print(f"\n💾 详细数据已保存: {csv_path}")
        print("\n📋 实验结果汇总:")
        print(df.to_string())

        # 过滤掉失败或跳过的实验
        df_plot = df[df['Status'] == 'Success']

        if len(df_plot) == 0:
            print("⚠️  没有成功的实验可绘图")
            return

        # 绘图
        plt.figure(figsize=(12, 8))

        # 不同密度的耗时对比
        densities = df_plot['Density'].unique()
        x = np.arange(len(densities))

        total_times = []
        compute_times = []
        gen_times = []

        for d in densities:
            data = df_plot[df_plot['Density'] == d]
            if len(data) > 0:
                total_times.append(data['Total Time (s)'].iloc[0])
                compute_times.append(data['Compute Time (s)'].iloc[0])
                gen_times.append(data['Gen Time (s)'].iloc[0])

        # 堆叠柱状图
        plt.bar(x, gen_times, label='数据生成', color='lightblue')
        plt.bar(x, compute_times, bottom=gen_times, label='矩阵计算', color='lightcoral') 

        plt.xlabel('矩阵密度')
        plt.ylabel('时间 (秒)')
        plt.title(f'Baseline矩阵乘法 ({algo_type}算法): 1000x1000矩阵')
        plt.xticks(x, [f'密度 {d}' for d in densities])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, (total, compute, gen) in enumerate(zip(total_times, compute_times, gen_times)):
            plt.text(i, total + max(total_times)*0.02, f'{total:.1f}s', ha='center')      
            plt.text(i, gen/2, f'{gen:.1f}s', ha='center', color='black')
            plt.text(i, gen + compute/2, f'{compute:.1f}s', ha='center', color='black')   

        plt.tight_layout()

        # 保存图片
        img_path = os.path.join(self.results_dir, f"baseline_{algo_type}_performance.png")        
        plt.savefig(img_path)
        print(f"📊 图表已生成: {img_path}")

        # 打印分析总结
        print("\n" + "="*80)
        print(f"📈 Baseline矩阵乘法实验分析总结 ({algo_type}算法):")
        print("="*80)

        for _, row in df_plot.iterrows():
            print(f"\n密度 {row['Density']}:")
            print(f"  总耗时: {row['Total Time (s)']:.2f}s")
            print(f"  计算时间: {row['Compute Time (s)']:.2f}s")
            print(f"  Shuffle Read: {row['Shuffle Read (MB)']:.2f}MB")
            print(f"  Shuffle Write: {row['Shuffle Write (MB)']:.2f}MB")
            print(f"  结果元素数: {row['Result Elements']}")

if __name__ == "__main__":
    use_optimized = True

    print("="*80)
    if use_optimized:
        print("运行优化版Baseline算法（使用join操作）")
    else:
        print("运行朴素版Baseline算法（使用Cartesian product）")      
    print("="*80)

    exp = BaselineMatrixExperiment()
    exp.run_experiment(use_optimized=use_optimized)
