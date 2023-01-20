'
A shell script to start the Triton server.
Fixed the path for triton server on /fsx
All moved the converted models are converted for 8 gpus
'
MODEL_PATH=${1}
singularity run --nv /fsx/elm_ft/triton_launch.sif/ mpirun -n 1 tritonserver --model-repository=${MODEL_PATH}
