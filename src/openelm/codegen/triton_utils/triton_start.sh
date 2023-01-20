:'
A shell script to start the Triton server.
Fixed the path for triton server on /fsx
All moved the converted models are converted for 8 gpus
'
singularity run --nv /fsx/elm_ft/mains.sif/ mpirun -n 1 tritonserver --model-repository=/fsx/elm_ft/codegen-350M-mono-8gpu/