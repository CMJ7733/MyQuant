#!/bin/sh
set -e
# 1. 检查条目是否已存在，如果不存在则添加
if ! grep -q "mirrors.baidubce.com" /etc/hosts; then
  echo "10.0.18.2 mirrors.baidubce.com" >> /etc/hosts
fi

echo "start task-server"
# 为了方便调试，暂时bos文件公开读，稳定后增加鉴权 or 存放到CFS上
if [ "$EXP_FRAME_VERSION" = "v1" ]; then
  TASK_SERVER_URL='https://provisioner.bj.bcebos.com/taskserver/fm-taskserver.tar.gz'
else
  TASK_SERVER_URL='https://provisioner.bj.bcebos.com/taskserver/fm-taskserver_v2.tar.gz'
fi
cd /tmp/ && mkdir fm-taskserver && cd fm-taskserver
curl --retry 5 --retry-all-errors --retry-delay 1 -fSL "$TASK_SERVER_URL" -o taskserver.tar.gz
tar xzvf taskserver.tar.gz && chmod +x bin/fm-taskserver
bin/fm-taskserver > /tmp/fm-taskserver/fm-taskserver.log 2>&1 &
echo "start task-server finish"

# echo "start code-server"
# nohup /data/code-server/bin/code-server \
#    --auth none \
#    --bind-addr 0.0.0.0:8087 \
#    --config /data/code-server/config.yaml \
#    --user-data-dir /data/code-server/user-data \
#    --extensions-dir /data/code-server/extensions \
#    /data/baidu/acgbenchmark/alpha_evolve/examples &
# echo "start code-server finish"
echo "start famou api"
cd /data/baidu/acg/fm-agent/api_server
nohup bash run.sh
echo "start famou api finished"