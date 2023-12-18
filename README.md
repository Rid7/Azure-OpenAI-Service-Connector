# Azure OpenAI Service Connector

Azure OpenAI 服务多节点连接服务

## 配置文件

config/config.yaml

格式：

```
AZURE_OPENAI:	# 固定名称
  api_type: "azure"	# 固定名称
  api_version: "2023-12-01-preview"	# 你的api版本
  regions:	# 你的不同区域的配置信息
      Japan:	# 假设你在日本有个节点，可以取名为Japan，方便区分节点
	api_base: "https://xxx.openai.azure.com/"	# 日本节点的api_base
      	available_models: [
        "gpt-4", "gpt-4-1106-preview", "gpt-35-turbo", "gpt-35-turbo-16k"]	# 你在该节点下部署的模型名称
      region2:	# 其他节点仿照日本节点配置
	...
```

## 主体逻辑

参见llm.azure_openai.py

## 测试

```
python -m unittest test.test_azure_openai
```
