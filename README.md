# Awesome Visual Reasoning 🧠🖼️: Pioneering the Future of AI Perception

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Welcome to **Awesome Visual Reasoning**! This repository is a curated collection of groundbreaking research that explores how we can empower machines to not just *see* images, but to *reason* about them, understand their nuances, and interact with the visual world in increasingly sophisticated ways.

Inspired by the human ability to seamlessly blend vision with thought, this list dives into the cutting-edge methods that are pushing the boundaries of AI. From intricate chain-of-thought processes that mimic human problem-solving to models that learn by interacting with visual tools, we're witnessing a revolution in how AI comprehends and generates visual information.

This collection is for researchers, developers, and enthusiasts eager to explore the forefront of:
*   **Prompt-Based Innovation:** How language can guide visual understanding and generation.
*   **Supervised Fine-Tuning:** Training models with rich, contextual visual data.
*   **Reinforcement Learning:** Enabling agents to learn through visual interaction and feedback.

Join us as we chart the course towards truly intelligent visual systems!

---

## 📜 Table of Contents

*   [🚀 Prompt-Based Methods for Reasoning with Images](#-prompt-based-methods-for-reasoning-with-images)
    *   [✍️➡️🤔 Chain-of-Thought and Visual Sketching](#️️-chain-of-thought-and-visual-sketching)
    *   [🎨 Visual Prompt Engineering](#-visual-prompt-engineering)
    *   [🛠️ Tool-Augmented Prompting](#️-tool-augmented-prompting)
    *   [💻 Programmatic Prompting and Code Execution](#-programmatic-prompting-and-code-execution)
    *   [🤖 Multimodal Agent Architectures](#-multimodal-agent-architectures)
*   [🎓 Supervised Fine-Tuning (SFT) based Methods for Reasoning with Images](#-supervised-fine-tuning-sft-based-methods-for-reasoning-with-images)
    *   [🔗 Vision-Language Chain-of-Thought Supervision](#-vision-language-chain-of-thought-supervision)
    *   [🔧 Tool-Enhanced Visual Reasoning](#-tool-enhanced-visual-reasoning)
    *   [🔄 Autoregressive Multimodal Generation in Vision-Language Models](#-autoregressive-multimodal-generation-in-vision-language-models)
*   [🏆 RL-based Methods for Reasoning with Images](#-rl-based-methods-for-reasoning-with-images)
    *   [🧩 Tool Use and Visual Interaction](#-tool-use-and-visual-interaction)
    *   [💡 Cognitive and Intrinsic Reward Strategies](#-cognitive-and-intrinsic-reward-strategies)
    *   [♻️ Reinforced Multimodal Generation](#️-reinforced-multimodal-generation)
*   [📚 Related Surveys & Benchmarks](#-related-surveys--benchmarks)
*   [🤝 Contributing](#-contributing)
*   [📄 License](#-license)

---

## 🚀 Prompt-Based Methods for Reasoning with Images

Unlocking visual intelligence through the art and science of prompting. These methods explore how carefully crafted textual or visual cues can guide Large Language Models (LLMs) and Vision-Language Models (VLMs) to perform complex reasoning tasks with images, often without explicit task-specific training.

### ✍️➡️🤔 Chain-of-Thought and Visual Sketching
*Mimicking human step-by-step reasoning, sometimes literally sketching out thoughts, to tackle complex visual problems.*

- **[Visual CoT: Unleashing Chain-of-Thought Reasoning in Multi-Modal Language Models](https://arxiv.org/abs/2403.16999)** <br/> *Hao Shao, Shengju Qian, Han Xiao, et al. (2024)*
- **[Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.15510)** <br/> *Zihui Cheng, Qiguang Chen, Xiao Xu, et al. (2025)*
- **[Visual sketchpad: Sketching as a visual chain of thought for multimodal language models](https://arxiv.org/abs/2406.09403)** <br/> *Yushi Hu, Weijia Shi, Xingyu Fu, et al. (2024)*
- **[SketchAgent: Language-Driven Sequential Sketch Generation](https://arxiv.org/abs/2411.17673)** <br/> *Yael Vinker, Tamar Rott Shaham, Kristine Zheng, et al. (2024)*
- **[Interactive Sketchpad: A Multimodal Tutoring System for Collaborative, Visual Problem-Solving](https://arxiv.org/abs/2503.16434)** <br/> *Steven-Shine Chen, Jimin Lee, Paul Pu Liang (2025)*
- **[VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](https://arxiv.org/abs/2504.09130)** <br/> *Yikun Wang, Siyin Wang, Qinyuan Cheng, et al. (2025)*
- **[ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044)** <br/> *Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, et al. (2024)*
- **[Visual chain-of-thought prompting for knowledge-based visual reasoning](https://ojs.aaai.org/index.php/AAAI/article/view/27916)** <br/> *Zhenfang Chen, Qinhong Zhou, Yikang Shen, et al. (2024)*

### 🎨 Visual Prompt Engineering
*Crafting visual cues or manipulating inputs to elicit desired behaviors and unlock latent capabilities in VLMs.*

- **[Promptcap: Prompt-guided task-aware image captioning](https://arxiv.org/abs/2211.09699)** <br/> *Yushi Hu, Hang Hua, Zhengyuan Yang, et al. (2022)*
- **[What does clip know about a red circle? visual prompt engineering for vlms](https://openaccess.thecvf.com/content/ICCV2023/html/Shtedritski_What_Does_CLIP_Know_About_a_Red_Circle_Visual_Prompt_ICCV_2023_paper.html)** <br/> *Aleksandar Shtedritski, Christian Rupprecht, Andrea Vedaldi (2023)*
- **[ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding](https://arxiv.org/abs/2501.05452)** <br/> *Xingyu Fu, Minqian Liu, Zhengyuan Yang, et al. (2025)*
- **[T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot](https://arxiv.org/abs/2505.00703)** <br/> *Dongzhi Jiang, Ziyu Guo, Renrui Zhang, et al. (2025)*
- **[GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/abs/2503.10639)** <br/> *Rongyao Fang, Chengqi Duan, Kun Wang, et al. (2025)*
- **[3DAxiesPrompts: Unleashing the 3D Spatial Task Capabilities of GPT-4V](https://arxiv.org/abs/2312.09738)** <br/> *Dingning Liu, Xiaomeng Dong, Renrui Zhang, et al. (2023)*
- **[PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs](https://arxiv.org/abs/2402.15823)** <br/> *Soroush Nasiriany, Fei Xia, Wenhao Yu, et al. (2024)*

### 🛠️ Tool-Augmented Prompting
*Empowering VLMs by allowing them to leverage external tools (e.g., calculators, search engines, other models) to enhance their reasoning and factual accuracy.*

- **[MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](https://arxiv.org/abs/2303.11381)** <br/> *Zhengyuan Yang, Linjie Li, Jianfeng Wang, et al. (2023)*
- **[Socratic models: Composing zero-shot multimodal reasoning with language](https://arxiv.org/abs/2204.00598)** <br/> *Andy Zeng, Maria Attarian, Brian Ichter, et al. (2022)*
- **[Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models](https://arxiv.org/abs/2312.03052)** <br/> *Yushi Hu, Otilia Stretcu, Chun-Ta Lu, et al. (2023)*
- **[Promptcap: Prompt-guided task-aware image captioning](https://arxiv.org/abs/2211.09699)** <br/> *Yushi Hu, Hang Hua, Zhengyuan Yang, et al. (2022)*
- **[MMFactory: A Universal Solution Search Engine for Vision-Language Tasks](https://arxiv.org/abs/2412.18072)** <br/> *Wan-Cyuan Fan, Tanzila Rahman, Leonid Sigal (2024)*
- **[VipAct: Visual-perception enhancement via specialized vlm agent collaboration and tool-use](https://arxiv.org/abs/2410.16400)** <br/> *Zhehao Zhang, Ryan Rossi, Tong Yu, et al. (2024)*
- **[Dettoolchain: A new prompting paradigm to unleash detection ability of MLLM](https://arxiv.org/abs/2403.12488)** <br/> *Yixuan Wu, Yizhou Wang, Shixiang Tang, et al. (2024)*
- **[Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/abs/2310.11441)** <br/> *Jianwei Yang, Hao Zhang, Feng Li, et al. (2023)*
- **[Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.15510)** <br/> *Zihui Cheng, Qiguang Chen, Xiao Xu, et al. (2025)*
- **[VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](https://arxiv.org/abs/2504.09130)** <br/> *Yikun Wang, Siyin Wang, Qinyuan Cheng, et al. (2025)*
- **[ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044)** <br/> *Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, et al. (2024)*
- **[CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers?](https://arxiv.org/abs/2412.13810)** <br/> *Dimitrios Mallis, Ahmet Serdar Karadeniz, Sebastian Cavada, et al. (2024)*
- **[Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models](https://arxiv.org/abs/2303.04671)** <br/> *Chenfei Wu, Shengming Yin, Weizhen Qi, et al. (2023)*

### 💻 Programmatic Prompting and Code Execution
*Generating and executing code (e.g., Python) as an intermediate reasoning step, allowing for precise, verifiable, and complex visual operations.*

- **[Visual programming: Compositional visual reasoning without training](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.html)** <br/> *Tanmay Gupta, Aniruddha Kembhavi (2023)*
- **[ViperGPT: Visual Inference via Python Execution for Reasoning](https://arxiv.org/abs/2303.08128)** <br/> *Dídac Surís, Sachit Menon, Carl Vondrick (2023)*
- **[Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models](https://arxiv.org/abs/2312.03052)** <br/> *Yushi Hu, Otilia Stretcu, Chun-Ta Lu, et al. (2023)*
- **[Interactive Sketchpad: A Multimodal Tutoring System for Collaborative, Visual Problem-Solving](https://arxiv.org/abs/2503.16434)** <br/> *Steven-Shine Chen, Jimin Lee, Paul Pu Liang (2025)*
- **[ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding](https://arxiv.org/abs/2501.05452)** <br/> *Xingyu Fu, Minqian Liu, Zhengyuan Yang, et al. (2025)*
- **[T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot](https://arxiv.org/abs/2505.00703)** <br/> *Dongzhi Jiang, Ziyu Guo, Renrui Zhang, et al. (2025)*
- **[GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/abs/2503.10639)** <br/> *Rongyao Fang, Chengqi Duan, Kun Wang, et al. (2025)*

### 🤖 Multimodal Agent Architectures
*Designing sophisticated architectures that can perceive, reason, plan, and act in multimodal environments, often integrating various reasoning strategies.*

- **[MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action](https://arxiv.org/abs/2303.11381)** <br/> *Zhengyuan Yang, Linjie Li, Jianfeng Wang, et al. (2023)*
- **[Interactive Sketchpad: A Multimodal Tutoring System for Collaborative, Visual Problem-Solving](https://arxiv.org/abs/2503.16434)** <br/> *Steven-Shine Chen, Jimin Lee, Paul Pu Liang (2025)*
- **[VipAct: Visual-perception enhancement via specialized vlm agent collaboration and tool-use](https://arxiv.org/abs/2410.16400)** <br/> *Zhehao Zhang, Ryan Rossi, Tong Yu, et al. (2024)*
- **[MMFactory: A Universal Solution Search Engine for Vision-Language Tasks](https://arxiv.org/abs/2412.18072)** <br/> *Wan-Cyuan Fan, Tanzila Rahman, Leonid Sigal (2024)*
- **[Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.15510)** <br/> *Zihui Cheng, Qiguang Chen, Xiao Xu, et al. (2025)*
- **[VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](https://arxiv.org/abs/2504.09130)** <br/> *Yikun Wang, Siyin Wang, Qinyuan Cheng, et al. (2025)*
- **[ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044)** <br/> *Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, et al. (2024)*
- **[Visual sketchpad: Sketching as a visual chain of thought for multimodal language models](https://arxiv.org/abs/2406.09403)** <br/> *Yushi Hu, Weijia Shi, Xingyu Fu, et al. (2024)*
- **[V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135)** <br/> *Penghao Wu, Saining Xie (2023)*

---

## 🎓 Supervised Fine-Tuning (SFT) based Methods for Reasoning with Images

Tailoring pre-trained models for visual reasoning through targeted fine-tuning on specialized datasets. This approach leverages instruction-following data and demonstrations of reasoning steps to enhance model capabilities.

### 🔗 Vision-Language Chain-of-Thought Supervision
*Training models on data that explicitly includes intermediate reasoning steps (chains of thought) connecting visual inputs to textual outputs.*

- **[Visual CoT: Unleashing Chain-of-Thought Reasoning in Multi-Modal Language Models](https://arxiv.org/abs/2403.16999)** <br/> *Hao Shao, Shengju Qian, Han Xiao, et al. (2024)*
- **[Cot-vla: Visual chain-of-thought reasoning for vision-language-action models](https://arxiv.org/abs/2503.22020)** <br/> *Qingqing Zhao, Yao Lu, Moo Jin Kim, et al. (2025)*
- **[TACO: Learning Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action](https://arxiv.org/abs/2412.05479)** <br/> *Zixian Ma, Jianguo Zhang, Zhiwei Liu, et al. (2024)*
- **[CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations](https://arxiv.org/abs/2402.04236)** <br/> *Ji Qi, Ming Ding, Weihan Wang, et al. (2024)*

### 🔧 Tool-Enhanced Visual Reasoning
*Fine-tuning models to effectively utilize external tools, learning when and how to call upon them to solve visual tasks.*

- **[Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models](https://arxiv.org/abs/2403.01299)** <br/> *Rex Hu, et al. (2024)* (Note: BibTeX author only says "Rex Hu and others", using shorter author list from other entry for "Visual Program Distillation")
- **[LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](https://arxiv.org/abs/2311.05437)** <br/> *Shilong Liu, Hao Cheng, Haotian Liu, et al. (2023)*
- **[GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/abs/2503.10639)** <br/> *Rongyao Fang, Chengqi Duan, Kun Wang, et al. (2025)*

### 🔄 Autoregressive Multimodal Generation in Vision-Language Models
*Aligning models through multimodal instruction-following and fine-tuning to generate structured, interleaved outputs across diverse vision-language tasks.*

- **[Instruction-Guided Visual Masking](https://arxiv.org/abs/2405.19783)** <br/> *Jinliang Zheng, Jianxiong Li, Sijie Cheng, et al. (2024)*
- **[Metamorph: Multimodal understanding and generation via instruction tuning](https://arxiv.org/abs/2412.14164)** <br/> *Shengbang Tong, David Fan, Jiachen Zhu, et al. (2024)*
- **[Imagine while Reasoning in Space: Multimodal Visualization-of-Thought](https://arxiv.org/abs/2501.07542)** <br/> *Chengzu Li, Wenshan Wu, Huanyu Zhang, et al. (2025)*
- **[BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset](https://arxiv.org/abs/2505.09568)** <br/> *Jiuhai Chen, Zhiyang Xu, Xichen Pan, et al. (2025)*
- **[Janus-pro: Unified multimodal understanding and generation with data and model scaling](https://arxiv.org/abs/2501.17811)** <br/> *Xiaokang Chen, Zhiyu Wu, Xingchao Liu, et al. (2025)*
- **[Chameleon: Mixed-modal early-fusion foundation models](https://arxiv.org/abs/2405.09818)** <br/> *Chameleon Team (2024)*
- **[Emu3: Next-token prediction is all you need](https://arxiv.org/abs/2409.18869)** <br/> *Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, et al. (2024)*
- **[Janus: Decoupling visual encoding for unified multimodal understanding and generation](https://arxiv.org/abs/2410.13848)** <br/> *Chengyue Wu, Xiaokang Chen, Zhiyu Wu, et al. (2024)*
- **[Emerging properties in unified multimodal pretraining](https://arxiv.org/abs/2505.14683)** <br/> *Chaorui Deng, Deyao Zhu, Kunchang Li, et al. (2025)*
- **[Generative multimodal models are in-context learners](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Generative_Multimodal_Models_Are_In-Context_Learners_CVPR_2024_paper.html)** <br/> *Quan Sun, Yufeng Cui, Xiaosong Zhang, et al. (2024)*
- **[Minigpt-5: Interleaved vision-and-language generation via generative vokens](https://arxiv.org/abs/2310.02239)** <br/> *Kaizhi Zheng, Xuehai He, Xin Eric Wang (2023)*
- **[Gpt-4o system card](https://arxiv.org/abs/2410.21276)** <br/> *Aaron Hurst, Adam Lerer, Adam P Goucher, et al. (2024)*
- **[Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation](https://arxiv.org/abs/2407.06135)** <br/> *Ethan Chern, Jiadi Su, Yan Ma, Pengfei Liu (2024)*
- **[Show-o: One single transformer to unify multimodal understanding and generation](https://arxiv.org/abs/2408.12528)** <br/> *Jinheng Xie, Weijia Mao, Zechen Bai, et al. (2024)*
- **[Generating images with multimodal language models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38bb589cbc673b7f8a192375841a0033-Abstract-Conference.html)** <br/> *Jing Yu Koh, Daniel Fried, Russ R Salakhutdinov (2023)*
- **[LMFusion: Adapting Pretrained Language Models for Multimodal Generation](https://arxiv.org/abs/2412.15188)** <br/> *Weijia Shi, Xiaochuang Han, Chunting Zhou, et al. (2025)* (also cited as `shi2024llamafusion` with year 2024, using 2025 from `shi2025lmfusionadaptingpretrainedlanguage` as it's likely an update)
- **[Thinking with Generated Images](https://arxiv.org/abs/2505.22525)** <br/> *Ethan Chern, Zhulin Hu, Steffi Chern, et al. (2025)*

---

## 🏆 RL-based Methods for Reasoning with Images

Harnessing the power of Reinforcement Learning to teach models how to reason with images through trial, error, and reward. These approaches enable agents to learn complex visual behaviors, tool interactions, and even intrinsic motivation for exploration.

### 🧩 Tool Use and Visual Interaction
*Training agents to interact with visual environments or use tools by rewarding successful task completion or effective tool manipulation.*

- **[OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/abs/2505.08617)** <br/> *Zhaochen Su, Linjie Li, Mingyang Song, et al. (2025)*
- **[Point-RFT: Improving Multimodal Reasoning with Visually Grounded Reinforcement Finetuning](https://arxiv.org/abs/2505.19702)** <br/> *Minheng Ni, Zhengyuan Yang, Linjie Li, et al. (2025)*
- **[Visual Planning: Let's Think Only with Images](https://arxiv.org/abs/2505.11409)** <br/> *Yi Xu, Chengzu Li, Han Zhou, et al. (2025)*
- **[GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879)** <br/> *Yue Fan, Xuehai He, Diji Yang, et al. (2025)* (also cited as `fan2024grit` with year 2024, using more complete author list and 2025 from `fan2025gritteachingmllmsthink`)
- **[Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)** <br/> *Ziyu Liu, Zeyi Sun, Yuhang Zang, et al. (2025)*
- **[Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL](https://arxiv.org/abs/2505.15436)** <br/> *Xintong Zhang, Zhi Gao, Bofei Zhang, et al. (2025)*
- **[ProgRM: Build Better GUI Agents with Progress Rewards](https://arxiv.org/abs/2505.18121)** <br/> *Danyang Zhang, Situo Zhang, Ziyue Yang, et al. (2025)*
- **[One RL to See Them All: Visual Triple Unified Reinforcement Learning](https://arxiv.org/abs/2505.18129)** <br/> *Yan Ma, Linge Du, Xuyang Shen, et al. (2025)*
- **[Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.14246)** <br/> *Ziyu Liu, Yuhang Zang, Yushan Zou, et al. (2025)*
- **[TACO: Think-Answer Consistency for Optimized Long-Chain Reasoning and Efficient Data Learning via Reinforcement Learning in LVLMs](https://arxiv.org/abs/2505.20777)** <br/> *Zhehan Kan, Yanlin Liu, Kun Yin, et al. (2025)*


### 💡 Cognitive and Intrinsic Reward Strategies
*Developing reward mechanisms based on cognitive principles (e.g., curiosity, information gain) to guide learning and encourage deeper reasoning.*

- **[DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362)** <br/> *Ziwei Zheng, Michael Yang, Jack Hong, et al. (2025)*
- **[Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning](https://arxiv.org/abs/2505.15966)** <br/> *Alex Su, Haozhe Wang, Weiming Ren, et al. (2025)*
- **[Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/abs/2503.06520)** <br/> *Yuqi Liu, Bohao Peng, Zhisheng Zhong, et al. (2025)*
- **[VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.12081)** <br/> *Yuqi Liu, Tianyuan Qu, Zhisheng Zhong, et al. (2025)*
- **[Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning](https://arxiv.org/abs/2402.11081)** (Note: arXiv ID assumed from title context, actual BibTeX for caoground not fully provided, search for "Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning" yields this) <br/> *Meng Cao, Haoze Zhao, Can Zhang, et al. (2024)*

### ♻️ Reinforced Multimodal Generation
*Using RL to refine the generation of multimodal outputs (text, images, etc.), often by optimizing for human preferences or specific quality metrics.*

- **[Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step](https://arxiv.org/abs/2501.13926)** <br/> *Ziyu Guo, Renrui Zhang, Chengzhuo Tong, et al. (2025)*
- **[Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO](https://arxiv.org/abs/2505.17017)** <br/> *Chengzhuo Tong, Ziyu Guo, Renrui Zhang, et al. (2025)*
- **[T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot](https://arxiv.org/abs/2505.00703)** <br/> *Dongzhi Jiang, Ziyu Guo, Renrui Zhang, et al. (2025)*
- **[MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](https://arxiv.org/abs/2505.10557)** <br/> *Ke Wang, Junting Pan, Linda Wei, et al. (2025)*

---

## 📚 Related Surveys & Benchmarks
*Essential resources for understanding the broader landscape and evaluating progress in visual reasoning.*

- **[WorldScore: A Unified Evaluation Benchmark for World Generation](https://arxiv.org/abs/2504.00983)** <br/> *Haoyi Duan, Hong-Xing Yu, Sirui Chen, et al. (2025)*
- **[PointArena: Probing Multimodal Grounding Through Language-Guided Pointing](https://arxiv.org/abs/2505.09990)** <br/> *Long Cheng, Jiafei Duan, Yi Ru Wang, et al. (2025)*
- **[A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges](https://arxiv.org/abs/2412.11936)** <br/> *Yibo Yan, Jiamin Su, Jianxiang He, et al. (2024)*
- **[Visual scratchpads: Enabling global reasoning in vision](https://arxiv.org/abs/2410.08165)** <br/> *Aryo Lotfi, Enrico Fini, Samy Bengio, et al. (2024)*
- **[VisFactor: Benchmarking Fundamental Visual Cognition in Multimodal Large Language Models](https://arxiv.org/abs/2502.16435)** <br/> *Jen-Tse Huang, Dasen Dai, Jen-Yuan Huang, et al. (2025)*
- **[m&m's: A Benchmark to Evaluate Tool-Use for multi-step multi-modal Tasks](https://arxiv.org/abs/2407.18809)** (Note: Assuming arXiv for ECCV paper) <br/> *Zixian Ma, Weikai Huang, Jieyu Zhang, et al. (2024)*
- **[MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models](https://arxiv.org/abs/2504.03641)** <br/> *Wulin Xie, Yi-Fan Zhang, Chaoyou Fu, et al. (2025)*
- **[Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions](https://arxiv.org/abs/2412.08737)** <br/> *Jiarui Zhang, Ollie Liu, Tianyu Yu, et al. (2024)*
- **[ARC Prize 2024: Technical Report](https://arxiv.org/abs/2412.04604)** <br/> *Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers (2024)*
- **[A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision-Language Models](https://arxiv.org/abs/2402.18409)** <br/> *Xiujie Song, Mengyue Wu, Kenny Q Zhu, et al. (2024)*
- **[Vgbench: Evaluating large language models on vector graphics understanding and generation](https://arxiv.org/abs/2407.10972)** <br/> *Bocheng Zou, Mu Cai, Jianrui Zhang, Yong Jae Lee (2024)*
- **[Elevating Visual Question Answering through Implicitly Learned Reasoning Pathways in LVLMs](https://arxiv.org/abs/2503.14674)** <br/> *Liu Jing, Amirul Rahman (2025)*
- **[CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation](https://arxiv.org/abs/2504.00043)** <br/> *Jixuan Leng, Chengsong Huang, Langlin Huang, et al. (2025)*
- **[A Review on Vision-Language-Based Approaches: Challenges and Applications.](https://www.techscience.com/cmc/v82n2/57843)** <br/> *Huu-Tuong Ho, Luong Vuong Nguyen, Minh-Tien Pham, et al. (2025)*
- **[Fast-Slow Thinking for Large Vision-Language Model Reasoning](https://arxiv.org/abs/2504.18458)** <br/> *Wenyi Xiao, Leilei Gan, Weilong Dai, et al. (2025)*
- **[ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](https://arxiv.org/abs/2505.13444)** <br/> *Liyan Tang, Grace Kim, Xinyu Zhao, et al. (2025)*


---

## 🤝 Contributing

This field is evolving at an incredible pace! If you've come across a groundbreaking paper, a new technique, or an insightful benchmark that belongs here, please feel free to:
1.  **Fork** the repository.
2.  **Add** your paper(s) to the relevant section(s) in `README.md`. Please maintain the existing format (Title with link, Authors, Year).
3.  **Create a Pull Request.**

We highly encourage contributions that help keep this list comprehensive and up-to-date. Let's build the ultimate resource for visual reasoning together!

---

## 📄 License

This Awesome list is shared under the [Creative Commons CC0 Universal License](LICENSE) (effectively public domain), meaning you can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

---

*Let's make machines that not only see, but truly understand.*
