## Guided Evolution:

<p align="left">
  <img src="./assets/header.png" alt="">
</p>
<br>

### Introduction:

In the ever-evolving domain of machine learning, the convergence of human cognitive skills and automated algorithms is entering a pivotal junction. This paper introduces “Guided Evolution” (GE), a novel framework that combines the human-like expertise of Large Language Models (LLMs) with the robust capabilities of Neural Architecture Search (NAS) through genetic algorithms. This innovative fusion advances automated machine learning, elevating traditional NAS by integrating a more insightful, intelligently guided evolutionary process.

Central to this framework is our  “Evolution of Thought” (EoT) technique, which extends and refines concepts like Zero-Shot Chain-of-Thought, Automated Chain-of-Thought, and Tree-of-Thought. These methodologies aim to improve the reasoning capabilities of LLMs. EoT takes a unique step forward by enabling LLMs to receive result-driven feedback, empowering them to make informed improvements based on the performance of their prior code augmentations, a significant advancement in intelligent automated machine learning.

EoT catalyzes LLMs to introspect and fine-tune suggestions based on past iterations, creating a self-enhancing feedback loop that fine-tunes architectural evolution. At the same, GE maintains essential genetic diversity for evolutionary algorithms while injecting human-like expertise and creativity into the evolutionary framework. Building from the insights of Ma et al., our Guided Evolutionary framework is further enhanced by a Character Role Play (CRP) technique, to markedly increase the feasibility, usefulness and creativity of ideas engendered by the LLM. 

The effectiveness of the Guided Evolution (GE) framework is showcased in the evolution of the ExquisiteNetV2 model. This evolution, initiated with a State-Of-The-Art (SOTA) seed model, not only demonstrates the capacity of LLMs to build upon and enhance SOTA models in collaboration with human expertise but also underscores their autonomous model design. This case study illustrates the framework's self-sufficient ability to generate improved model variants, emphasizing the burgeoning impact of LLMs in redefining traditional model design pipelines, a step towards models that independently evolve and refine their architectures. 

______

### Setup

This code utilizes [ExquisiteNetV2](https://github.com/shyhyawJou/ExquisiteNetV2) which is copied into the sota directory.

Dependencies are managed through `pyproject.toml`. This package can be installed with `pip install .` or interacted with through tools such as `uv` 

Then follow the instructions to prepare the CIFAR10 dataset in the [ExquisiteNetV2 README](./sota/ExquisiteNetV2/README.md)

This code has been tested on Python 3.12

If you wish to use features with Google's Gemini, please follow the instructions for [Setting Up an API Key](https://ai.google.dev/gemini-api/docs/api-key)

______

### Autonomous Model Evolution:

<p align="center">
  <img src="./assets/ge_run1.gif" alt="">
</p>

______

### Paper

[LLM Guided Evolution - The Automation of Models Advancing Models](./assets/paper/LLM_Guided_Evolution___The_Automation_of_Models_Advancing_Models.pdf)

______

### Cited by DeepMind **AlphaEvolve**

DeepMind’s Gemini-powered agent **AlphaEvolve** references our *LLM-Guided Evolution* framework as prior work (see reference \[72\] in the white-paper). 

AlphaEvolve shares the GE concept of letting an LLM propose, mutate, and test code and scales it to discover state-of-the-art algorithms across mathematics, scheduling, and hardware design.

We’re proud that the concepts first open-sourced here are now being implemented in frontier research at Google DeepMind.  
Read the white-paper below:

<p align="left">
  <a href="https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/dc/Google_DeepMind_logo.svg" alt="DeepMind logo" width="250" style="height:auto;">
  </a>
</p>
<br>


