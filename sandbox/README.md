# ELM sandboxing:  

For now, we have a sandboxed server that we can hit up from outside the sandbox. At the moment, it only returns the strings we pass it. The goal is for the server to 
1. receive Python encodings for Sodaracers as strings, 
2. run them in the Sodaracer environment once it is ready, 
3. return relevant information (loss, reward, anything else?).

## Getting started
- [Install gVisor](https://gvisor.dev/docs/user_guide/install/)
- Make sure you have [docker](https://docs.docker.com/get-docker/) installed
- `pip install --user pipenv` for package management

## Working on the server
- Run `scripts/build.sh` whenever you make changes to server code. 
- Run `scripts/launch.sh` to launch the server in a sandboxed container.


## Tasks

- [X] run Linux on gVisor locally  
- [X] run python in a container locally  
- [X] run python in gVisor  
- [X] python server in gVisor taking requests from outside gVisor  
- [ ] sodaracer environment in gVisor  
    - [pyleSOR](https://github.com/dmahan93/pyIesorPhysics)  
    - [Sodaracers Python Encoding](https://github.com/CarperAI/ELM/pull/3), maybe we can restrict exec to the relevant functions?  
- [ ] gVisor with firewall rules  

