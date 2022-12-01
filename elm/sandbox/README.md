# ELM sandboxing:

For now, we have a sandboxed server that we can hit up from outside the sandbox with code and it will return a serialized walker object. e.g.
1. receive Python encodings for Walkers as strings,
2. generate the Walker in the sandboxed server,
3. return the serialized walker to the caller.

Evaluation can in principle be done outside of the sandbox since the arbitrary code is only used for generating walklers, which are then safe.

## Getting started
- [Install gVisor](https://gvisor.dev/docs/user_guide/install/)
- Make sure you have [docker](https://docs.docker.com/get-docker/) installed
- `pip install --user pipenv` for package management
- `pipenv shell`
- `pipenv install`
- `sudo runsc install`

## Working on the server
- Start docker - `service docker start`
- Run `scripts/build.sh` whenever you make changes to server code.
- Run `scripts/launch.sh` to launch the server in a sandboxed container.


## Tasks

- [X] run Linux on gVisor locally
- [X] run python in a container locally
- [X] run python in gVisor
- [X] python server in gVisor taking requests from outside gVisor
- [X] sodaracer environment in gVisor
    - [pyleSOR](https://github.com/dmahan93/pyIesorPhysics)
    - [Sodaracers Python Encoding](https://github.com/CarperAI/ELM/pull/3), maybe we can restrict exec to the relevant functions?
- [X] generate walkers in server
