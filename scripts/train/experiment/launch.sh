# #!/bin/bash
# # set -ex

# Check for singlenode flag
SCRIPT=$1
SINGLENODE=${SINGLENODE:-false}

if [ "$SINGLENODE" == "true" ]; then
    bash $SCRIPT
    exit 0
fi

#==============================================================================#

unset ROCR_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export NCCL_IB_TIMEOUT=80
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_AR_THRESHOLD=0

export MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
export MASTER_PORT=${MLP_WORKER_0_PORT:-10086}
export NNODES=${MLP_WORKER_NUM:-1}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-0}
export GPUS_PER_NODE=${MLP_WORKER_GPU:-8}

# Compute total world size (number of processes)
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"

export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=$PWD:$PYTHONPATH

# num_nodes has to be at least 1
if [ $NNODES -lt 1 ]; then
    echo "Number of nodes must be at least 1"
    exit 1
fi

# if HOST contains "master", then this is the head node
if [[ $NODE_RANK -eq 0 ]]; then
    node_role="master"
else
    node_role="worker"
fi
head_node_ip=${MLP_WORKER_0_PRIMARY_HOST:-127.0.0.1}

wait_time=30
if [ "$node_role" == "master" ]; then
    echo "Starting Ray head node..."
    # Start Ray on this node as the head node and extract its address
    # The `ray start --head` command outputs information that includes the address,
    # but here we're assuming it's known or statically assigned for simplicity.
    ray start --head --node-ip-address=$head_node_ip --include-dashboard=True --dashboard-host=0.0.0.0 --port=6379
    sleep $wait_time
elif [ "$node_role" == "worker" ]; then
    sleep $wait_time
    attempt=1
    echo "Starting Ray worker node and attempting to connect to the head node at $head_node_ip:6379"
    while true; do
        # Attempt to start Ray and connect to the head node
        ray start --address="$head_node_ip:6379" && break || {
            if [ $attempt -le 5 ]; then
                echo "Ray worker start attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        }
    done
fi

# run the training script once Ray has been started on all nodes
sleep $wait_time
if [ "$node_role" == "master" ]; then
    num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
    echo "Number of active Ray nodes: $num_active_ray_nodes"
    if [ $num_active_ray_nodes -lt $NNODES ]; then
        echo "Waiting for all Ray nodes to start..."
        attempt=1
        while true; do
            num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
            if [ $num_active_ray_nodes -eq $NNODES ]; then
                break
            elif [ $attempt -le 5 ]; then
                echo "python command attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        done
    fi
    echo "End starting"
    bash ${SCRIPT}
else
    echo "End starting"
    # Continuously check the health of the Ray cluster by pinging the head node.
    # If the health check fails, break the loop and proceed.
    while true; do
        ray health-check --address $head_node_ip:6379 &>/dev/null
        if [ $? -ne 0 ]; then
            break
        fi
        sleep 60
    done
fi
