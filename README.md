## Guided Evolution:

![Logo](https://github.com/MosesTheRedSea/LLM-Guided-Evolution-Generic/blob/main/llm-guided-evolution-image.png)

https://i.ibb.co/whMScMY/Picture1.png

______

### Introduction:
______

In the ever-evolving domain of machine learning, the convergence of human cognitive skills and automated algorithms is entering a pivotal junction. This paper introduces “Guided Evolution” (GE), a novel framework that combines the human-like expertise of Large Language Models (LLMs) with the robust capabilities of Neural Architecture Search (NAS) through genetic algorithms. This innovative fusion advances automated machine learning, elevating traditional NAS by integrating a more insightful, intelligently guided evolutionary process.

Central to this framework is our  “Evolution of Thought” (EoT) technique, which extends and refines concepts like Zero-Shot Chain-of-Thought, Automated Chain-of-Thought, and Tree-of-Thought \cite{kojima2022large, zhang2022automatic, yao2023tree}. These methodologies aim to improve the reasoning capabilities of LLMs. EoT takes a unique step forward by enabling LLMs to receive result-driven feedback, empowering them to make informed improvements based on the performance of their prior code augmentations, a significant advancement in intelligent automated machine learning.

EoT catalyzes LLMs to introspect and fine-tune suggestions based on past iterations, creating a self-enhancing feedback loop that fine-tunes architectural evolution. At the same, GE maintains essential genetic diversity for evolutionary algorithms while injecting human-like expertise and creativity into the evolutionary framework. Building from the insights of Ma et al. \cite{ma2023conceptual}, our Guided Evolutionary framework is further enhanced by a Character Role Play (CRP) technique, to markedly increase the feasibility, usefulness and creativity of ideas engendered by the LLM. 

The effectiveness of the Guided Evolution (GE) framework is showcased in the evolution of the ExquisiteNetV2 model. This evolution, initiated with a State-Of-The-Art (SOTA) seed model, not only demonstrates the capacity of LLMs to build upon and enhance SOTA models in collaboration with human expertise but also underscores their autonomous model design. This case study illustrates the framework's self-sufficient ability to generate improved model variants, emphasizing the burgeoning impact of LLMs in redefining traditional model design pipelines, a step towards models that independently evolve and refine their architectures. 

### Setup

This code utilizes [ExquisiteNetV2](https://github.com/shyhyawJou/ExquisiteNetV2) which is copied into the sota directory.

First install dependencies using `pip install -r requirements.txt`

Then follow the instructions to prepare the CIFAR10 dataset in the [ExquisiteNetV2 README](./sota/ExquisiteNetV2/README.md)

This code has been tested on Python 3.12

### Autonomous Model Evolution:
_____
<p align="center">
  <img src="./assets/ge_run1.gif" alt="">
</p>

### Paper
_____

[LLM Guided Evolution - The Automation of Models Advancing Models](./assets/paper/LLM_Guided_Evolution___The_Automation_of_Models_Advancing_Models.pdf)
