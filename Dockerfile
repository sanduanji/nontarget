FROM registry.cn-hangzhou.aliyuncs.com/ijcai_competition/env:env
MAINTAINER cutrain <duanyuge@qq.com>
RUN conda install numpy
COPY . /competition
WORKDIR /competition
