# Docker 部署指南

## 前提条件

- 已安装 [Docker](https://docs.docker.com/get-docker/)
- 已安装 [Docker Compose](https://docs.docker.com/compose/install/)

## 快速开始

### 1. 配置环境变量

在运行前，请先配置项目根目录下的 `.env` 文件。将示例值替换为你自己的 API 密钥：

```bash
# 编辑 .env 文件
vi .env
```

### 2. 构建并启动容器

```bash
# 构建镜像并启动容器
docker-compose up --build
```

此命令将：
- 构建包含 Python 3.11 和 Poetry 的 Docker 镜像
- 安装所有项目依赖
- 使用默认参数运行应用程序

### 3. 使用自定义参数运行

如果要使用自定义参数运行程序，可以通过以下方式：

```bash
# 覆盖默认命令
docker-compose run ai-hedge-fund --tickers "TSLA,NVDA,AMZN" --start-date "2023-01-01" --end-date "2023-12-31" --show-reasoning
```

## 可用参数

```
--initial-cash          初始现金金额 (默认: 100000.0)
--margin-requirement    初始保证金要求 (默认: 0.0)
--tickers               股票代码列表 (必需，逗号分隔)
--start-date            开始日期 (YYYY-MM-DD格式)
--end-date              结束日期 (YYYY-MM-DD格式)
--show-reasoning        显示每个代理的推理过程
--show-agent-graph      显示代理图
```

## 交互式使用

程序将会询问你选择哪些AI分析师和LLM模型。通过Docker运行时，你可以在终端中进行交互。

## 故障排除

### 如果遇到代理连接问题

如果你在构建镜像时遇到类似以下错误：
```
Could not connect to 127.0.0.1:7890 (127.0.0.1). - connect (111: Connection refused)
```

请尝试以下方法：

**方法1：临时关闭代理**
```bash
# 临时关闭所有代理环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 然后再构建镜像
docker build -t ai-hedge-fund -f src/dockerfile .
```

**方法2：使用不需要Python 3.11的简化版Dockerfile**
```bash
# 使用简化版Dockerfile
docker build -t ai-hedge-fund -f src/dockerfile.no-proxy .
```

**方法3：设置Docker守护进程不使用代理**
编辑或创建Docker守护进程配置文件：
```bash
sudo mkdir -p /etc/systemd/system/docker.service.d/
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
```

确保文件内容为（完全清除代理设置）：
```
[Service]
Environment="HTTP_PROXY="
Environment="HTTPS_PROXY="
Environment="NO_PROXY=*"
```

然后重启Docker服务：
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 如果遇到版本兼容性问题

如果你看到类似以下错误：
```
ERROR: Version in "./docker-compose.yml" is unsupported. You might be seeing this error because you're using the wrong Compose file version.
```

请尝试以下解决方法：
```bash
# 检查你的Docker Compose版本
docker-compose --version

# 如果版本较低，请使用以下命令安装最新版本
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

或者直接降低`docker-compose.yml`文件中的版本号（已完成此修改）。

### 如果遇到权限问题

```bash
# 授权当前用户对.env文件的访问权限
chmod 600 .env
```

### 如果需要查看容器日志

```bash
# 查看容器日志
docker-compose logs
```

### 如果需要进入容器内部

```bash
# 进入正在运行的容器
docker-compose exec ai-hedge-fund bash
```

## 配置 Docker 镜像加速器

如果你在中国大陆或某些网络环境中遇到镜像拉取失败的问题，可以配置Docker使用国内的镜像加速器来加速镜像拉取。

1. 创建或修改 `daemon.json` 文件：

```bash
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
```

在文件中添加以下内容（可以选择其中一个或多个）：

```json
{
  "registry-mirrors": [
    "https://registry.docker-cn.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

保存文件后重启Docker服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

然后再次尝试构建：

```bash
docker build -t ai-hedge-fund -f src/dockerfile .
```

或者在构建时使用代理：

```bash
docker build --build-arg HTTP_PROXY=http://127.0.0.1:7890 --build-arg HTTPS_PROXY=http://127.0.0.1:7890 -t ai-hedge-fund -f src/dockerfile .
```

或者使用本地基础镜像：

```bash
docker build --build-arg BASE_IMAGE=python:3.11-slim -t ai-hedge-fund -f src/dockerfile .
``` 

apt update

apt-get install -y     vim     curl     wget     git     nano     htop     net-tools     iputils-ping     dnsutils     sudo     less     procps     zip     unzip     tar     ca-certificates     lsb-release     gnupg     apt-transport-https     && apt-get clean     && rm -rf /var/lib/apt/lists/*



unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy
unset NO_PROXY

git config --global --unset-all http.proxy

git config --global --unset-all https.proxy