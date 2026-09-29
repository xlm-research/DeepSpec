export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
# 节点之间的通信直接连接，绕过 HTTP 代理
export no_proxy="${no_proxy},10.120.0.211,10.120.0.218,10.120.0.187"
export NO_PROXY="$no_proxy"