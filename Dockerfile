FROM registry.cn-hangzhou.aliyuncs.com/ijcai_competition/env:env
MAINTAINER cutrain <duanyuge@qq.com>
COPY ./ijcai_nontarget_attack /competition
WORKDIR /competition
