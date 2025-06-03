# Thinking with Images for LVLMs: A Survey

[![Awesome](https://awesome.re/badge.svg)](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zhaochen0110/Awesome_Think_With_Images?color=green) 

</div>

---

## 🔔 News

- [2025-06] We created this repository to maintain a paper list on Awesome-Think-With-Images.
  
Welcome to **Thinking with Images for LVLMs: A Survey**! This repository serves as a curated and comprehensive overview of pivotal research dedicated to enabling Large Vision-Language Models (LVLMs) to truly **think with images**. We delve into how these sophisticated models are evolving beyond mere pattern recognition, acquiring capabilities for intricate reasoning, nuanced understanding, and dynamic interaction by processing and interpreting visual information in cognitive-inspired ways.

This collection is for researchers, developers, and enthusiasts eager to explore the forefront of:
*   **Prompt-Based Innovation:** How LVLMs can guide visual understanding and generation.
*   **Supervised Fine-Tuning:** Training models with rich, contextual visual data.
*   **Reinforcement Learning:** Enabling agents to learn through visual interaction and feedback.

Join us as we chart the course towards truly intelligent visual systems!

---

## 📜 Table of Contents

*   [🚀 Prompt-Based Methods for Reasoning with Images](#-prompt-based-methods-for-reasoning-with-images)
*   [🎓 Supervised Fine-Tuning (SFT) based Methods for Reasoning with Images](#-supervised-fine-tuning-sft-based-methods-for-reasoning-with-images)
*   [🏆 RL-based Methods for Reasoning with Images](#-rl-based-methods-for-reasoning-with-images)
*   [📚 Related Surveys & Benchmarks](#-related-surveys--benchmarks)
*   [🤝 Contributing](#-contributing)
*   [📄 License](#-license)

---

## 🚀 Prompt-Based Methods for Reasoning with Images

Unlocking visual intelligence through the art and science of prompting. These methods explore how carefully crafted textual or visual cues can guide LVLMs to perform complex reasoning tasks with images, often without explicit task-specific training.

### ✍️➡️🤔 Chain-of-Thought and Visual Sketching
*Mimicking human step-by-step reasoning, sometimes literally sketching out thoughts, to tackle complex visual problems.*

- [Visual CoT: Unleashing Chain-of-Thought Reasoning in Multi-Modal Language Models](https://arxiv.org/abs/2403.16999) ![](https://img.shields.io/badge/abs-2024.03-red)
- [Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.15510) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Visual sketchpad: Sketching as a visual chain of thought for multimodal language models](https://arxiv.org/abs/2406.09403) ![](https://img.shields.io/badge/abs-2024.06-red)
- [SketchAgent: Language-Driven Sequential Sketch Generation](https://arxiv.org/abs/2411.17673) ![](https://img.shields.io/badge/abs-2024.11-red)
- [Interactive Sketchpad: A Multimodal Tutoring System for Collaborative, Visual Problem-Solving](https://arxiv.org/abs/2503.16434) ![](https://img.shields.io/badge/abs-2025.03-red)
- [VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](https://arxiv.org/abs/2504.09130) ![](https://img.shields.io/badge/abs-2025.04-red)
- [ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044) ![](https://img.shields.io/badge/abs-2024.11-red)
- [ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding](https://arxiv.org/abs/2501.05452) ![](https://img.shields.io/badge/abs-2025.01-red)

### 🛠️ Tool-Augmented Prompting
*Empowering VLMs by allowing them to leverage external tools (e.g., calculators, search engines, other models) to enhance their reasoning and factual accuracy.*

- [MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](https://arxiv.org/abs/2303.11381) ![](https://img.shields.io/badge/abs-2023.03-red)
- [Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models](https://arxiv.org/abs/2303.04671) ![](https://img.shields.io/badge/abs-2023.03-red)
- [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/abs/2310.11441) ![](https://img.shields.io/badge/abs-2023.10-red)
- [MMFactory: A Universal Solution Search Engine for Vision-Language Tasks](https://arxiv.org/abs/2412.18072) ![](https://img.shields.io/badge/abs-2024.12-red)
- [VipAct: Visual-perception enhancement via specialized vlm agent collaboration and tool-use](https://arxiv.org/abs/2410.16400) ![](https://img.shields.io/badge/abs-2024.10-red)
- [Dettoolchain: A new prompting paradigm to unleash detection ability of MLLM](https://arxiv.org/abs/2403.12488) ![](https://img.shields.io/badge/abs-2024.03-red)

- [Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.15510) ![](https://img.shields.io/badge/abs-2025.05-red)
- [VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](https://arxiv.org/abs/2504.09130) ![](https://img.shields.io/badge/abs-2025.04-red)
- [ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044) ![](https://img.shields.io/badge/abs-2024.11-red)

- [CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers?](https://arxiv.org/abs/2412.13810) ![](https://img.shields.io/badge/abs-2024.12-red)


### 💻 Programmatic Prompting and Code Execution
*Generating and executing code (e.g., Python) as an intermediate reasoning step, allowing for precise, verifiable, and complex visual operations.*

- [Visual programming: Compositional visual reasoning without training](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.html) ![](https://img.shields.io/badge/CVPR-2023-blue)
- [ViperGPT: Visual Inference via Python Execution for Reasoning](https://arxiv.org/abs/2303.08128) ![](https://img.shields.io/badge/ICCV-2023-blue) <!-- Note: Original bib was ICCV, arXiv link available -->
- [Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models](https://arxiv.org/abs/2312.03052) ![](https://img.shields.io/badge/abs-2023.12-red)
- [Interactive Sketchpad: A Multimodal Tutoring System for Collaborative, Visual Problem-Solving](https://arxiv.org/abs/2503.16434) ![](https://img.shields.io/badge/abs-2025.03-red)
- [ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding](https://arxiv.org/abs/2501.05452) ![](https://img.shields.io/badge/abs-2025.01-red)
- [T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot](https://arxiv.org/abs/2505.00703) ![](https://img.shields.io/badge/abs-2025.05-red)
- [GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/abs/2503.10639) ![](https://img.shields.io/badge/abs-2025.03-red)

### 🤖 Multimodal Agent Architectures
*Designing sophisticated architectures that can perceive, reason, plan, and act in multimodal environments, often integrating various reasoning strategies.*

- [MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](https://arxiv.org/abs/2303.11381) ![](https://img.shields.io/badge/abs-2023.03-red)
- [Interactive Sketchpad: A Multimodal Tutoring System for Collaborative, Visual Problem-Solving](https://arxiv.org/abs/2503.16434) ![](https://img.shields.io/badge/abs-2025.03-red)
- [VipAct: Visual-perception enhancement via specialized vlm agent collaboration and tool-use](https://arxiv.org/abs/2410.16400) ![](https://img.shields.io/badge/abs-2024.10-red)
- [MMFactory: A Universal Solution Search Engine for Vision-Language Tasks](https://arxiv.org/abs/2412.18072) ![](https://img.shields.io/badge/abs-2024.12-red)
- [ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044) ![](https://img.shields.io/badge/abs-2024.11-red)
- [Visual sketchpad: Sketching as a visual chain of thought for multimodal language models](https://arxiv.org/abs/2406.09403) ![](https://img.shields.io/badge/abs-2024.06-red)
- [V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135) ![](https://img.shields.io/badge/abs-2023.12-red)

- [T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot](https://arxiv.org/abs/2505.00703) ![](https://img.shields.io/badge/abs-2025.05-red)
- [GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/abs/2503.10639) ![](https://img.shields.io/badge/abs-2025.03-red)
---

## 🎓 Supervised Fine-Tuning (SFT) based Methods for Reasoning with Images

Tailoring pre-trained models for visual reasoning through targeted fine-tuning on specialized datasets. This approach leverages instruction-following data and demonstrations of reasoning steps to enhance model capabilities.

### 🔗 Vision-Language Chain-of-Thought Supervision
*Training models on data that explicitly includes intermediate reasoning steps (chains of thought) connecting visual inputs to textual outputs.*

- [Visual CoT: Unleashing Chain-of-Thought Reasoning in Multi-Modal Language Models](https://arxiv.org/abs/2403.16999) ![](https://img.shields.io/badge/abs-2024.03-red)
- [Cot-vla: Visual chain-of-thought reasoning for vision-language-action models](https://arxiv.org/abs/2503.22020) ![](https://img.shields.io/badge/abs-2025.03-red)
- [TACO: Learning Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action](https://arxiv.org/abs/2412.05479) ![](https://img.shields.io/badge/abs-2024.12-red)
- [CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations](https://arxiv.org/abs/2402.04236) ![](https://img.shields.io/badge/abs-2024.02-red)

### 🔧 Tool-Enhanced Visual Reasoning
*Fine-tuning models to effectively utilize external tools, learning when and how to call upon them to solve visual tasks.*

- [Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models](https://arxiv.org/abs/2403.01299) ![](https://img.shields.io/badge/abs-2024.03-red)
- [LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](https://arxiv.org/abs/2311.05437) ![](https://img.shields.io/badge/abs-2023.11-red)
- [GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/abs/2503.10639) ![](https://img.shields.io/badge/abs-2025.03-red)

### 🔄 Autoregressive Multimodal Generation in Vision-Language Models
*Aligning models through multimodal instruction-following and fine-tuning to generate structured, interleaved outputs across diverse vision-language tasks.*

- [Instruction-Guided Visual Masking](https://arxiv.org/abs/2405.19783) ![](https://img.shields.io/badge/abs-2024.05-red)
- [Metamorph: Multimodal understanding and generation via instruction tuning](https://arxiv.org/abs/2412.14164) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Imagine while Reasoning in Space: Multimodal Visualization-of-Thought](https://arxiv.org/abs/2501.07542) ![](https://img.shields.io/badge/abs-2025.01-red)
- [BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset](https://arxiv.org/abs/2505.09568) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Janus-pro: Unified multimodal understanding and generation with data and model scaling](https://arxiv.org/abs/2501.17811) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Chameleon: Mixed-modal early-fusion foundation models](https://arxiv.org/abs/2405.09818) ![](https://img.shields.io/badge/abs-2024.05-red)
- [Emu3: Next-token prediction is all you need](https://arxiv.org/abs/2409.18869) ![](https://img.shields.io/badge/abs-2024.09-red)
- [Janus: Decoupling visual encoding for unified multimodal understanding and generation](https://arxiv.org/abs/2410.13848) ![](https://img.shields.io/badge/abs-2024.10-red)
- [Emerging properties in unified multimodal pretraining](https://arxiv.org/abs/2505.14683) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Generative multimodal models are in-context learners](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Generative_Multimodal_Models_Are_In-Context_Learners_CVPR_2024_paper.html) ![](https://img.shields.io/badge/CVPR-2024-blue)
- [Minigpt-5: Interleaved vision-and-language generation via generative vokens](https://arxiv.org/abs/2310.02239) ![](https://img.shields.io/badge/abs-2023.10-red)
- [Gpt-4o system card](https://arxiv.org/abs/2410.21276) ![](https://img.shields.io/badge/abs-2024.10-red)
- [Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation](https://arxiv.org/abs/2407.06135) ![](https://img.shields.io/badge/abs-2024.07-red)
- [Show-o: One single transformer to unify multimodal understanding and generation](https://arxiv.org/abs/2408.12528) ![](https://img.shields.io/badge/abs-2024.08-red)
- [Generating images with multimodal language models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38bb589cbc673b7f8a192375841a0033-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
- [LMFusion: Adapting Pretrained Language Models for Multimodal Generation](https://arxiv.org/abs/2412.15188) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Thinking with Generated Images](https://arxiv.org/abs/2505.22525) ![](https://img.shields.io/badge/abs-2025.05-red)

---

## 🏆 RL-based Methods for Reasoning with Images

Harnessing the power of Reinforcement Learning to teach models how to reason with images through trial, error, and reward. These approaches enable agents to learn complex visual behaviors, tool interactions, and even intrinsic motivation for exploration.

### 🧩 Tool Use and Visual Interaction
*Training agents to interact with visual environments or use tools by rewarding successful task completion or effective tool manipulation.*

- [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/abs/2505.08617) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Point-RFT: Improving Multimodal Reasoning with Visually Grounded Reinforcement Finetuning](https://arxiv.org/abs/2505.19702) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Visual Planning: Let's Think Only with Images](https://arxiv.org/abs/2505.11409) ![](https://img.shields.io/badge/abs-2025.05-red)
- [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785) ![](https://img.shields.io/badge/abs-2025.03-red)
- [Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL](https://arxiv.org/abs/2505.15436) ![](https://img.shields.io/badge/abs-2025.05-red)
- [ProgRM: Build Better GUI Agents with Progress Rewards](https://arxiv.org/abs/2505.18121) ![](https://img.shields.io/badge/abs-2025.05-red)
- [One RL to See Them All: Visual Triple Unified Reinforcement Learning](https://arxiv.org/abs/2505.18129) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.14246) ![](https://img.shields.io/badge/abs-2025.05-red)
- [TACO: Think-Answer Consistency for Optimized Long-Chain Reasoning and Efficient Data Learning via Reinforcement Learning in LVLMs](https://arxiv.org/abs/2505.20777) ![](https://img.shields.io/badge/abs-2025.05-red)


### 💡 Cognitive and Intrinsic Reward Strategies
*Developing reward mechanisms based on cognitive principles (e.g., curiosity, information gain) to guide learning and encourage deeper reasoning.*

- [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning](https://arxiv.org/abs/2505.15966) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/abs/2503.06520) ![](https://img.shields.io/badge/abs-2025.03-red)
- [VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.12081) ![](https://img.shields.io/badge/abs-2025.05-red)
- [Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning](https://arxiv.org/abs/2402.11081) ![](https://img.shields.io/badge/abs-2024.02-red)

### ♻️ Reinforced Multimodal Generation
*Using RL to refine the generation of multimodal outputs (text, images, etc.), often by optimizing for human preferences or specific quality metrics.*

- [Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step](https://arxiv.org/abs/2501.13926) ![](https://img.shields.io/badge/abs-2025.01-red)
- [Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO](https://arxiv.org/abs/2505.17017) ![](https://img.shields.io/badge/abs-2025.05-red)
- [T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot](https://arxiv.org/abs/2505.00703) ![](https://img.shields.io/badge/abs-2025.05-red)
- [MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](https://arxiv.org/abs/2505.10557) ![](https://img.shields.io/badge/abs-2025.05-red)

---

## 📚 Related Surveys & Benchmarks
*Essential resources for understanding the broader landscape and evaluating progress in visual reasoning.*

- [WorldScore: A Unified Evaluation Benchmark for World Generation](https://arxiv.org/abs/2504.00983) ![](https://img.shields.io/badge/abs-2025.04-red)
- [PointArena: Probing Multimodal Grounding Through Language-Guided Pointing](https://arxiv.org/abs/2505.09990) ![](https://img.shields.io/badge/abs-2025.05-red)
- [A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges](https://arxiv.org/abs/2412.11936) ![](https://img.shields.io/badge/abs-2024.12-red)
- [Visual scratchpads: Enabling global reasoning in vision](https://arxiv.org/abs/2410.08165) ![](https://img.shields.io/badge/abs-2024.10-red)
- [VisFactor: Benchmarking Fundamental Visual Cognition in Multimodal Large Language Models](https://arxiv.org/abs/2502.16435) ![](https://img.shields.io/badge/abs-2025.02-red)
- [m&m's: A Benchmark to Evaluate Tool-Use for multi-step multi-modal Tasks](https://arxiv.org/abs/2407.18809) ![](https://img.shields.io/badge/ECCV-2024-blue)
- [MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models](https://arxiv.org/abs/2504.03641) ![](https://img.shields.io/badge/abs-2025.04-red)
- [Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions](https://arxiv.org/abs/2412.08737) ![](https://img.shields.io/badge/abs-2024.12-red)
- [ARC Prize 2024: Technical Report](https://arxiv.org/abs/2412.04604) ![](https://img.shields.io/badge/abs-2024.12-red)
- [A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision-Language Models](https://arxiv.org/abs/2402.18409) ![](https://img.shields.io/badge/abs-2024.02-red)
- [Vgbench: Evaluating large language models on vector graphics understanding and generation](https://arxiv.org/abs/2407.10972) ![](https://img.shields.io/badge/abs-2024.07-red)
- [Elevating Visual Question Answering through Implicitly Learned Reasoning Pathways in LVLMs](https://arxiv.org/abs/2503.14674) ![](https://img.shields.io/badge/abs-2025.03-red)
- [CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation](https://arxiv.org/abs/2504.00043) ![](https://img.shields.io/badge/abs-2025.04-red)
- [A Review on Vision-Language-Based Approaches: Challenges and Applications.](https://www.techscience.com/cmc/v82n2/57843) ![](https://img.shields.io/badge/CMC-2025-blue)
- [Fast-Slow Thinking for Large Vision-Language Model Reasoning](https://arxiv.org/abs/2504.18458) ![](https://img.shields.io/badge/abs-2025.04-red)
- [ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](https://arxiv.org/abs/2505.13444) ![](https://img.shields.io/badge/abs-2025.05-red)



---

<!-- ## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zhaochen0110/Awesome_Think_With_Images&type=Date)](https://star-history.com/#zhaochen0110/Awesome_Think_With_Images&Date) -->
*Let's make machines that not only see, but truly understand.*
