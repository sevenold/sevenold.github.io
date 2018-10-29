---
layout: post
title: "Docker镜像管理"
date: 2018-10-29
description: "Docker安装及系列教程"
tag: Docker
---

### Docker镜像管理

Docker镜像是一个不包含Linux内核而又精简的Linux的操作系统。

### Dock镜像下载

Docker Hub 是由Docker公司负责维护的公共注册中心，包含大量的容器镜像，Docker工具默认从这个公共镜像库下载镜像：https://hub.docker.com/explore

默认是国外的源，下载会很慢，可以使用国内的源提供下载速度：参考上一节[镜像加速器]()

### 镜像的工作原理

当我们启动一个新的容器时，Docker会加载只读镜像，并在其之上添加一个读写层，并将镜像中的目录复制一份到/var/lib/docker/aufs/mnt/容器ID为目录下，我们可以使用chroot进入此目录。

如果运行中的容器修改一个已经存在的文件，那么会将该文件从下面的只读层复制到读写层，只读层的这个文件就会覆盖，但还存在，这就实现了文件系统隔离，当删除容器后，读写层的数据将会删除，只读镜像不变。

### 镜像的文件存储结构

docker相关文件存放在：/var/lib/docker目录下

```bash
/var/lib/docker/aufs/diff # 每层与其父层之间的文件差异
/var/lib/docker/aufs/layers/ # 每层一个文件，记录其父层一直到根层之间的ID，大部分文件的最后一行都已，表示继承来自同一层
/var/lib/docker/aufs/mnt # 联合挂载点，从只读层复制到最上层可读写层的文件系统数据
```

在建立镜像时，每次写操作，都被视作一种增量操作，即在原有的数据层上添加一个新层；所以一个镜像会有若干个层组成。每次commit提交就会对产生一个ID，就相当于在上一层有加了一层，可以通过这个ID对镜像回滚。

### 镜像管理命令

列举一些常用的Docker镜像管理命令，完整内容请参考[官方文档](https://docs.docker.com/get-started/#docker-concepts)

### Docker search 命令

**docker search :** 从Docker Hub查找镜像

#### 语法：

```bash
docker search [OPTIONS] TERM
```

> OPTIONS说明：
>
> **--automated :**只列出 automated build类型的镜像；
>
> **--no-trunc :**显示完整的镜像描述；
>
> **-s : ** 列出star数不小于指定值的镜像。



#### 实例

从Docker Hub 查找所有镜像包含MySQL， 并且star大于100

```bash
seven@seven:~$ sudo docker search mysql -s 100
Flag --stars has been deprecated, use --filter=stars=3 instead
NAME                         DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
mysql                        MySQL is a widely used, open-source relation…   7213                [OK]                
mariadb                      MariaDB is a community-developed fork of MyS…   2311                [OK]                
mysql/mysql-server           Optimized MySQL Server Docker images. Create…   531                                     [OK]
percona                      Percona Server is a fork of the MySQL relati…   382                 [OK]                
zabbix/zabbix-server-mysql   Zabbix Server with MySQL database support       136                                     [OK]
```

### Docker pull 命令

**docker pull :** 从镜像仓库中拉取或者更新指定镜像

#### 语法：

```basn
docker pull [OPTIONS] NAME[:TAG|@DIGEST]
```

> OPTIONS说明：
>
> **-a :**拉取所有 tagged 镜像
>
> **--disable-content-trust :**忽略镜像的校验,默认开启

#### 实例：

从Docker Hub下载mysql。

```bash
seven@seven:~$ sudo docker pull mysql
Using default tag: latest
latest: Pulling from library/mysql
f17d81b4b692: Pull complete 
c691115e6ae9: Pull complete 
41544cb19235: Pull complete 
254d04f5f66d: Pull complete 
4fe240edfdc9: Pull complete 
0cd4fcc94b67: Pull complete 
8df36ec4b34a: Pull complete 
720bf9851f6a: Pull complete 
e933e0a4fddf: Pull complete 
9ffdbf5f677f: Pull complete 
a403e1df0389: Pull complete 
4669c5f285a6: Pull complete 
Digest: sha256:811483efcd38de17d93193b4b4bc4ba290a931215c4c8512cbff624e5967a7dd
Status: Downloaded newer image for mysql:latest
```

### Docker push 命令

**docker push :** 将本地的镜像上传到镜像仓库,要先登陆到镜像仓库

#### 语法：

```bash
docker push [OPTIONS] NAME[:TAG]
```

> OPTIONS说明：
>
> **--disable-content-trust :**忽略镜像的校验,默认开启



### Docker images 命令

Docker images: 查看本机的镜像

#### 语法：

```
docker images [OPTIONS] [REPOSITORY[:TAG]]
```

> OPTIONS说明：
>
> **-a :**列出本地所有的镜像（含中间映像层，默认情况下，过滤掉中间映像层）；
>
>
>
> **--digests :**显示镜像的摘要信息；
>
>
>
> **-f :**显示满足条件的镜像；
>
>
>
> **--format :**指定返回值的模板文件；
>
>
>
> **--no-trunc :**显示完整的镜像信息；
>
>
>
> **-q :**只显示镜像ID。

### 实例：

查看本地镜像列表

```bash
seven@seven:~$ sudo docker images 
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mysql               latest              2dd01afbe8df        4 days ago          485MB
ubuntu              latest              ea4c82dcd15a        10 days ago         85.8MB
hello-world         latest              4ab4c602aa5e        7 weeks ago         1.84kB
```

### Docker rmi 命令

**docker rmi: **删除本地一个或多少镜像。

#### 语法：

```bash
docker rmi [OPTIONS] IMAGE [IMAGE...]
```

> OPTIONS说明：
>
> **-f :**强制删除；
>
> **--no-prune :**不移除该镜像的过程镜像，默认移除；

### Docker export 命令

**docker export :**将文件系统作为一个tar归档文件导出到STDOUT。

### 语法

```bash
docker export [OPTIONS] CONTAINER
```

>  OPTIONS说明：
>
> **-o :**将输入内容写到文件。

### Docker import 命令

**docker import :** 从归档文件中创建镜像。

语法

```
docker import [OPTIONS] file|URL|- [REPOSITORY[:TAG]]
```

> OPTIONS说明：
>
> **-c :**应用docker 指令创建镜像；
>
> **-m :**提交时的说明文字；

### Docker save 命令

**docker save :** 将指定镜像保存成 tar 归档文件。

#### 语法

```
docker save [OPTIONS] IMAGE [IMAGE...]
```

> OPTIONS说明：
>
> **-o :**输出到的文件。

### Docker load命令

docker load:从归档文件中导入镜像。

#### 语法：

```bash
docker load [OPTIONS]
```

> Options:
>   -i: 从tar存档文件读取的输入字符串，而不是STDIN
>   -q:  Suppress the load output

