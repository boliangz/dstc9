# DiDi AI Lab's submission for DSTC9 Multi-domain Task-oriented Dialog Challenge II 

We participated in the [Multi-domain Task-oriented Dialog Challenge II](https://www.microsoft.com/en-us/research/project/multi-domain-task-completion-dialog-challenge-ii/) of [DSTC9](https://dstc9.dstc.community/home).
Participants in this shared task build end-to-end dialog systems that can assist human to fulfil single or multiple tasks, such as making a restaurant reservation, booking a hotel, etc. 

In the evaluation, real human users are recruited and chat with the system to fulfill tasks. At the end, human users determine whether the task is fulfilled based on system responses. 
Our best system achieves 74.8% success rate and ties for first place in the challenge. It scores 4.51/5 on language understanding and 4.45/5 on response appropriateness, which are the two additional metrics judged by human users.

We release our code for research purpose. 
The code is based on [Convlab-2](https://github.com/thu-coai/ConvLab-2) which is a dialog evaluation toolkit provided by the shared task organizer. 
Please also checkout our paper for more details: [A Hybrid Task-Oriented Dialog System with Domain and Task Adaptive Pretraining](https://drive.google.com/file/d/1GWZhY05C7aiiJZ9GE8smwME0V1X95M1h/view).

## Installation

1. Python version >= 3.5 
(A Python virtual environment is strongly recommended)

2. Install dependencies and download pre-trained models:
```bash
bash install.sh
```

3. Add the project root directory to `PYTHONPATH`:
```bash
export PYTHONPATH=<project_root_path>
```

## Quick Start

We provide a simple script that can let you interact with the chatbot in termial.

```bash
python end2end/human_eval_example.py
```

A randomly generated goal is provided, you could chat with the system by following the goal.
Type `success` or `fail` to finish the conversation.

## Evaluate with user simulator

Convlab-2 comes with a user simulator that can be used to evaluate chatbots. Please find more details about the user simulator on [Convlab-2 github page](https://github.com/thu-coai/ConvLab-2).

To run the user simulator based automatic evaluation:
```
python end2end/submission1/automatic.py
```

## Cite

Please cite this paper if you find this repository useful:
```
coming soon
```

## Contact

Questions? Suggestions? Please use Github "Issues" panel or email [boliangzhang@didiglobal.com](mailto:boliangzhang@didiglobal.com).   






