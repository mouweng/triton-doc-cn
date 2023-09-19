# 使用Triton部署你训练的模型

给定一个训练好的模型，如何使用Triton推理服务以最佳的配置来大规模地进行部署。这篇文档就是来回答这个问题。

- 想要查看精简的概述，下面的例子为通用流程。
- 想要直接上手，请跳转到下方端到端示例。
- 更多的细节，请参考[Triton概念指南教程](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_4-inference_acceleration)。

## 概述

**问题1: 我的模型可以在Triton上部署吗？**

- 如果你的模型在Triton支持的[Backend](https://github.com/triton-inference-server/backend)列表里，那么可以像[快速开始](用户指南/00-快速开始)指南中描述的那样，尝试部署一下模型。对于ONNXRuntime、TensorFlow、SavedModel和TensorRT的Backend，基本的模型配置可以使用Triton的[Autocomplete](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#auto-generated-model-configuration)功能从模型中推断出来。你仍然可以配置config.pbtxt但这并不是必须的，除非你想要显示的设置某些参数。另外，可以通过设置`--log-verbose=1`启用详细的Triton日志，你可以在Triton的日志输出中看到Triton内部的完整配置。对于其他的Backend，请参考[文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#minimal-model-configuration)。
- 如果你的模型不在Triton支持的Backend列表里，您可以查看[Python Backend](https://github.com/triton-inference-server/python_backend)或编写[Custom C++ Backend](https://github.com/triton-inference-server/backend/blob/main/examples/README.md)来支持您的模型。Python Backend提供了一个简单的接口，可以使用通用的Python脚本执行请求，但性能可能不如Custom C++ Backend。根据你自己的用例需求，Python Backend可能是在性能和简单实现性上的一个折衷选项。

**问题2: 我如何在部署的模型上进行推理?**

- 假设你能够正确的使用Triton加载模型，下一步是验证模型的性能。Triton的[Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)就可以用来做这件事情。下面是它的一个简单的输出效果：

```shell
# NOTE: "my_model" represents a model currently being served by Triton
$ perf_analyzer -m my_model
...

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 482.8 infer/sec, latency 12613 usec
```

- 这为我们提供了一个完整的测试闭环，我们可以使用Triton API对请求模型并接收相应，达到与模型Backend通信的目的。
- 如果Perf Analyzer发送请求失败，并且不清楚错误产生的原因，那么您可能需要对模型配置进行完整性检查。pbtxt的输入输出需要与模型期望的输入输出保持一致。如果配置是完全正确的，则检查模型是否可以通过其原始框架运行成功。如果你没有自己的脚本活着工具，[Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)可以在各种框架上面运行示例来判断，目前Polygraphy支持ONNXRuntime, TensorRT和TensorFlow 1.x。
- 对于不同的例子，其性能表现可能会不同。一些常见的指标有吞吐量、延迟和GPU利用率。可以在模型配置 (`config.pbtxt`) 中调整参数以获取最佳的配置。
- [Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)是验证模型功能和性能的最好工具。

**问题3: 我如何提高模型的性能?**



## 其他你可能感兴趣的地方

1. **我的模型第一次被Triton加载时执行缓慢(冷启动)，我该怎么办?**



2. **为什么我的模型在GPU上的运行速度没有显著提高?**



## 端到端示例

> ⚠️ 如果你之前没使用过Triton，你可能会对快速入门的示例感兴趣。下面的示例比较简单，可以让你对Triton有一个初步的了解，让你没有任何经验的情况下上手Triton。

让我们以ONNX模型为例，因为ONNX是一种绝大多数框架都可以导出的通用格式。

1. 创建一个模型仓库，并将我们的示例模型`densenet_onnx`下载到仓库中。

```shell
# Create model repository with placeholder for model and version 1
mkdir -p ./models/densenet_onnx/1

# Download model and place it in model repository
wget -O models/densenet_onnx/1/model.onnx
https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
```

2. 在模型仓库的``./models/densenet_onnx/config.pbtxt`路径下面为`densenet_onnx`模型创建一个最基本的模型配置。

> ⚠️ 这是一个简化版本的配置，本示例中暂时不需要使用到其他的配置参数。

```protobuf
name: "densenet_onnx"
backend: "onnxruntime"
max_batch_size: 0
input: [
  {
    name: "data_0",
    data_type: TYPE_FP32,
    dims: [ 1, 3, 224, 224]
  }
]
output: [
  {
    name: "prob_1",
    data_type: TYPE_FP32,
    dims: [ 1, 1000, 1, 1 ]
  }
]
```

> ⚠️ 从22.07版本开始，Triton和Model Analyzer可以不需要后端配置文件。因此，对于ONNX模型，除非你想显式设置某些参数，否则可以跳过此步骤。

3. 启动Triton容器

我们可以使用容器中预装好的tritonserver程序来启动我们的模型服务。

```shell
# Start server container
docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt --name triton-server nvcr.io/nvidia/tritonserver:23.08-py3

# Start serving your models
tritonserver --model-repository=/mnt/models
```

> ⚠️ `$PWD:/mnt`是将主机上的当前目录挂载到容器内的`/mnt`目录。因此，如果你在`$PWD:/mnt`中创建了模型仓库，那么你将会在`/mnt/models`的容器中找到它。你可以根据需要来更改这些路径。有关其工作原理的更多信息，请参考[Docker Volumes](https://docs.docker.com/storage/volumes/)文档。

如果模型加载成功，我们将会在输出的日志中看到我们的模型处于`READY`状态：

```shell
...
I0802 18:11:47.100537 135 model_repository_manager.cc:1345] successfully loaded 'densenet_onnx' version 1
...
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| densenet_onnx | 1       | READY  |
+---------------+---------+--------+
...
```

5. 验证模型是否可以进行推理

为了验证我们的模型可以执行推理，我们将使用`triton-client`容器，该容器预装了名为`perf_analyzer`的压测工具。

再另外开一个shell，我们可以使用`perf_analyzer`来检查模型是否可以进行推理，并可以测算我们期望的模型性能。

在下面的示例中，`perf_analyzer`将发送请求到同一台机器上的模型（因为指定了`--network=host`所以可以发送请求到`localhost`）。当然，你也可以通过添加`-u <IP>:<PORT>`参数来指定具体的主机和端口，例如`perf_analyzer -m densenet_onnx -u 127.0.0.1:8000`。

```shell
# Start the SDK container interactively
docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt --name triton-client nvcr.io/nvidia/tritonserver:23.08-py3-sdk

# Benchmark model being served from step 3
perf_analyzer -m densenet_onnx --concurrency-range 1:4
```

```shell
...
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 265.147 infer/sec, latency 3769 usec
Concurrency: 2, throughput: 890.793 infer/sec, latency 2243 usec
Concurrency: 3, throughput: 937.036 infer/sec, latency 3199 usec
Concurrency: 4, throughput: 965.21 infer/sec, latency 4142 usec
```

5. 运行Model Analyzer以查找模型的最佳配置
