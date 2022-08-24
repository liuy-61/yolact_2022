#基础镜像为 ubuntu，版本为20.04.1，build镜像时会自动下载
FROM ubuntu:20.04.1

#制作者信息
MAINTAINER liuyao@nuaa.edu.cn

#设置环境变量
ENV CODE_DIR=/home/liuy
ENV DOCKER_SCRIPTS=$CODE_DIR/base_image/scripts

#将scripts下的文件复制到镜像中的DOCKER_SCRIPTS目录
COPY ./scripts/* $DOCKER_SCRIPTS/

#执行镜像中的provision.sh脚本
RUN chmod a+x $DOCKER_SCRIPTS/*
RUN $DOCKER_SCRIPTS/provision.sh