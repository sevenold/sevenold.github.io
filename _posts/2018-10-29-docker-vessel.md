---
layout: post
title: "Docker容器管理"
date: 2018-10-29
description: "Docker安装及系列教程"
tag: Docker
---

### 启动容器

启动容器有两种方式，一种是基于镜像新建一个容器并启动，另外一个是将在终止状态（`stopped`）的容器重新启动。

### 新建并启动

所需要的命令主要为 `docker run`。

```bash
seven@seven:~$ sudo docker run -it ubuntu
root@c964215eb3c2:/# 
```

其中，`-t` 选项让Docker分配一个伪终端（pseudo-tty）并绑定到容器的标准输入上， `-i` 则让容器的标准输入保持打开。

在交互模式下，用户可以通过所创建的终端来输入命令，例如

```bash
root@c964215eb3c2:/# ls
bin  boot  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
root@c964215eb3c2:/# dir
bin  boot  dev	etc  home  lib	lib64  media  mnt  opt	proc  root  run  sbin  srv  sys  tmp  usr  var
root@c964215eb3c2:/# pwd
/
```

当利用 `docker run` 来创建容器时，Docker 在后台运行的标准操作包括：

- 检查本地是否存在指定的镜像，不存在就从公有仓库下载
- 利用镜像创建并启动一个容器
- 分配一个文件系统，并在只读的镜像层外面挂载一层可读写层
- 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
- 从地址池配置一个 ip 地址给容器
- 执行用户指定的应用程序
- 执行完毕后容器被终止

### 启动已终止容器

所需要的命令主要为：`docker container start`

```bash
seven@seven:~$ sudo docker container start -i  trusting_boyd
root@c964215eb3c2:/# 
```

其中： `-i` 则让容器的标准输入保持打开。

### 终止容器

可以使用 `docker container stop` 来终止一个运行中的容器。

此外，当 Docker 容器中指定的应用终结时，容器也自动终止。

当只启动了一个终端的容器，用户通过 `exit` 命令或 `Ctrl+d` 来退出终端时，所创建的容器立刻终止。

终止状态的容器可以用 `docker container ls -a` 命令看到。例如

```bash
seven@seven:~$ sudo docker container ls -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                          PORTS               NAMES
c964215eb3c2        ubuntu              "/bin/bash"         12 minutes ago      Exited (0) About a minute ago                       trusting_boyd
```

处于终止状态的容器，可以通过 `docker container start` 命令来重新启动。

此外，`docker container restart` 命令会将一个运行态的容器终止，然后再重新启动它。

### 进入容器

在使用 `-d` 参数时，容器启动后会进入后台。

某些时候需要进入容器进行操作，包括使用 `docker attach` 命令或 `docker exec` 命令，推荐大家使用 `docker exec` 命令，原因会在下面说明。

### `attach` 命令

`docker attach` 是 Docker 自带的命令。下面示例如何使用该命令。

```bash
seven@seven:~$ sudo docker run -dit ubuntu
aaa535c389754179db273c48bafbe8b2c514cb0699f5e77b1249394715965f5f
seven@seven:~$ sudo docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
aaa535c38975        ubuntu              "/bin/bash"         19 seconds ago      Up 17 seconds                           vigilant_cori
seven@seven:~$ sudo docker attach vigilant_cori
root@aaa535c38975:/# 
```

或者进入我们刚创建那个容器：

```bash
seven@seven:~$ sudo docker container start trusting_boyd
trusting_boyd
seven@seven:~$ sudo docker attach trusting_boyd
root@c964215eb3c2:/# 
```

### `exec` 命令

#### -i -t 参数

`docker exec` 后边可以跟多个参数，这里主要说明 `-i` `-t` 参数。

只用 `-i` 参数时，由于没有分配伪终端，界面没有我们熟悉的 Linux 命令提示符，但命令执行结果仍然可以返回。

当 `-i` `-t` 参数一起使用时，则可以看到我们熟悉的 Linux 命令提示符。

```bash
seven@seven:~$ sudo docker exec -i trusting_boyd  ls 
bin
boot
dev
etc
home
lib
lib64
media
mnt
opt
proc
root
run
sbin
srv
sys
tmp
usr
var
```

### 导出容器

如果要导出本地某个容器，可以使用 `docker export` 命令。

```bash
seven@seven:~/demo$ sudo docker export trusting_boyd > ubuntu.tar
seven@seven:~/demo$ ls
ubuntu.tar
```

这样将导出容器快照到本地文件。

### 导入容器

可以使用 `docker import` 从容器快照文件中再导入为镜像，例如

```bash
seven@seven:~/demo$ cat ubuntu.tar | sudo docker import - test/ubuntu:v1.0
sha256:2d78a6138d933a64fef619687dfb31f4323f607e35c980d1e70ec5a79a7cb39f
seven@seven:~/demo$ sudo docker images 
REPOSITORY          TAG                 IMAGE ID            CREATED              SIZE
test/ubuntu         v1.0                2d78a6138d93        About a minute ago   69.8MB
mysql               latest              2dd01afbe8df        4 days ago           485MB
ubuntu              latest              ea4c82dcd15a        10 days ago          85.8MB
hello-world         latest              4ab4c602aa5e        7 weeks ago          1.84kB
```

此外，也可以通过指定 URL 或者某个目录来导入，例如

```bash
$ docker import http://example.com/exampleimage.tgz example/imagerepo
```

*注：用户既可以使用 docker load 来导入镜像存储文件到本地镜像库，也可以使用 docker import 来导入一个容器快照到本地镜像库。这两者的区别在于容器快照文件将丢弃所有的历史记录和元数据信息（即仅保存容器当时的快照状态），而镜像存储文件将保存完整记录，体积也要大。此外，从容器快照文件导入时可以重新指定标签等元数据信息。*

### 删除容器

可以使用 `docker container rm` 来删除一个处于终止状态的容器。例如

```bash
$ docker container rm  trusting_newton
trusting_newton
```

如果要删除一个运行中的容器，可以添加 `-f` 参数。Docker 会发送 `SIGKILL` 信号给容器。

### 清理所有容器

```bash
seven@seven:~/demo$  sudo docker rm -f $(sudo docker ps -q -a)
aaa535c38975
c964215eb3c2
```

### 容器安装软件

就比如安装vim软件， 在使用docker容器时，有时候里边没有安装vim，敲vim命令时提示说：vim: command not found，这个时候就需要安装vim：

```bash
$ apt-get install vim
 Reading package lists... Done
 Building dependency tree       
 Reading state information... Done
 E: Unable to locate package vim
```

解决这个问题的方法：执行更新命令

```bash
$ apt-get update
```

这个命令的作用是：**同步 /etc/apt/sources.list 和 /etc/apt/sources.list.d 中列出的源的索引**，这样才能获取到最新的软件包。

等更新完毕以后再敲命令：

```bash
root@4c7345f38eab:~/.pip# apt-get install vim
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following additional packages will be installed:
  libgpm2 vim-common vim-runtime xxd
Suggested packages:
  gpm ctags vim-doc vim-scripts
The following NEW packages will be installed:
  libgpm2 vim vim-common vim-runtime xxd
0 upgraded, 5 newly installed, 0 to remove and 1 not upgraded.
Need to get 6766 kB of archives.
After this operation, 31.2 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
...
```

接下来你就可以在docker容器里面，安装你所需要的软件了。

