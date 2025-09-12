# TPU-Manual

Contacts: For any questions regarding the setup or usage of the TPU, please contact Boya Zeng or Taiming Lu via email, Slack, or Messenger. Please make sure to send a message on Messenger (instead of only sending a friend request on Facebook) for us to see it.

The content of this manual is composed by [Boya Zeng](https://boyazeng.github.io), [Yufeng Xu](https://zephyr271828.github.io), and [Taiming Lu](taiminglu.com).

## Overview
This manual is a guide for using the Google Cloud TPU in Zhuang's group at Princeton University.

TPUs are specialized high-performance computing resources optimized for large-scale machine learning workloads. We have access to multiple TPU generations and topologies, including v3, v4, v5e, and v6e pods, with both preemptible and on-demand configurations available depending on quota and project access.


## Table of Contents

- [TPU-Manual](#tpu-manual)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [TPU introduction](#tpu-introduction)
    - [Workflow](#workflow)
    - [TPU Specifications](#tpu-specifications)
      - [TPU Hardware Specifications](#tpu-hardware-specifications)
      - [Performance Benchmarks (Llama2 7B)](#performance-benchmarks-llama2-7b)


## TPU introduction

This section provides essential terminology and concepts you need to understand when working with Google Cloud TPUs. These definitions will help you navigate the TPU ecosystem and effectively utilize the available resources for your machine learning projects.


- **TPU (Tensor Processing Unit)**: Google’s hardware for fast machine learning (like a GPU, but optimized for deep learning).
- **TPU VM**: A virtual machine that you log into to run code directly on TPUs. Each TPU VM is associated with one or more TPUs, and you can request them from Google Cloud.
- **TPU Pod**: A cluster of many TPUs connected together with high-speed interconnects for large-scale distributed training.
- **Host**: The CPU machine (TPU VM) that manages and communicates with TPUs. In a pod configuration, there are multiple hosts, each managing a subset of the TPUs.
- **External IP**: The public IP address used to SSH into TPU VMs from your local computer.
- **Internal IP**: Private IP addresses used for communication between TPUs and hosts within the same pod or network.
- **Bucket**: Google Cloud Storage (GCS) containers where you store datasets, logs, and checkpoints. Note you only have 100G disk space on each TPU VM (host), so it's necessary to store your large files in a bucket first, then access it by mounting to your VM or by using [gsutil](https://cloud.google.com/storage/docs/gsutil).
- **SSH Key**: Cryptographic keys required for secure SSH connections to TPU VMs and for inter-host communication in multi-host distributed training jobs.
- **Region**: A large geographic area (e.g., us-central1, europe-west4) containing Google Cloud resources. _Buckets are tied to a specific region, meaning data is physically stored there_.
- **Zone**: A smaller location within a region (e.g., us-central1-b). _TPUs are created in zones_.

```
                ┌──────────────────────────────────────────────┐
                │                 REGION                       │
                │              (e.g., us-central1)             │
                │                                              │
                │  ┌────────────────┐     ┌─────────────────┐  │
GCS Bucket ───▶ │  │  GCS BUCKET    │     │     ZONE A      │  │
 (lives at      │  │ (your bucket)  │     │  ┌──────────┐   │  │
                │  └────────────────┘     │  │ TPU VM 1 │   │  │
                │                         │  └──────────┘   │  │
                │                         │  ┌──────────┐   │  │
                │                         │  │ TPU VM 2 │   │  │
                │                         │  └──────────┘   │  │
                │                         └─────────────────┘  │
                │                                              │
                │                         ┌─────────────────┐  │
                │                         │     ZONE B      │  │
                │                         │ (us-central1-b) │  │
                │                         │  ┌──────────┐   │  │
                │                         │  │ TPU VM 3 │   │  │
                │                         │  └──────────┘   │  │
                │                         └─────────────────┘  │
                └──────────────────────────────────────────────┘
```

### Workflow

The usual workflow of using TPUs is: 
1. Request a TPU VM with the desired TPU type and configuration through Google Cloud Console or gcloud CLI, 
2. SSH into the TPU VM using your external IP and SSH key and control TPUs using your local machine, 
3. Set up your environment and install necessary dependencies,
4. Upload your datasets and code to a GCS bucket or mount the bucket to your VM, 
5. Run your machine learning training jobs using frameworks like JAX, PyTorch XLA, or TensorFlow that are optimized for TPUs
6. Monitor your training progress and save checkpoints to the GCS bucket for persistence.

### TPU Specifications

The TPU versions we have access to are v2, v3, v4, v5e, and v6e. Understanding the specifications of each TPU generation is crucial for selecting the right hardware for your workload and optimizing your training jobs.

#### TPU Hardware Specifications

| TPU version | #core per chip | #process per chip | TPU memory per process | #chips per worker (vk-8 and above) |
|-------------|----------------|-------------------|------------------------|-------------------------------------|
| v3          | 2              | 2                 | 16 GB                  | 4                                   |
| v4          | 2              | 1                 | 32 GB                  | 4                                   |
| v5e         | 1              | 1                 | 16 GB                  | 4                                   |
| v6e         | 1              | 1                 | 32 GB                  | 4                                   |

**Key specifications to consider:**
- **Memory per process**: Determines the maximum model size you can train on a single TPU core
- **Cores per chip**: Affects parallelization within a single TPU chip
- **Processes per chip**: Number of independent training processes that can run simultaneously
- **Chips per worker**: For multi-chip configurations (vk-8 and above), this determines the total compute power

#### Performance Benchmarks (Llama2 7B)

| Device    | Tokens/s | Tokens/h | Tokens/day | Comparison |
|-----------|----------|----------|------------|------------|
| L40 (GPU) | 1,146    | 4.1M     | 0.10B      | 45%        |
| A100 (GPU)| 2,570    | 9.3M     | 0.22B      | 100%       |
| H100 (GPU)| 3,855    | 13M      | 0.33B      | 150%       |
| v2-1 (TPU)| 305      | 1.1M     | 0.03B      | 12%        |
| v3-1 (TPU)| 608      | 2.2M     | 0.05B      | 24%        |
| v4-1 (TPU)| 1,079    | 3.9M     | 0.09B      | 42%        |
| v5-1 (TPU)| 1,021    | 3.7M     | 0.08B      | 40%        |
| v6-1 (TPU)| 3,627    | 13M      | 0.31B      | 141%       |

*Note: Numbers reported are per GPU chip/TPU core. A100 is used as the 100% baseline for comparison.*


- **v6e TPUs** offer competitive performance, reaching 141% of A100 performance
- **v4 and v5e** provide moderate performance suitable for many workloads
- **v2 and v3** are older generations with limited performance for modern LLM training
- Choose TPU version based on your model size, training time requirements, and cost considerations

