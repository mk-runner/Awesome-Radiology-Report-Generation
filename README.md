# Awesome-Radiology-Report-Generation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

We collect existing papers on radiology report generation published in prominent conferences and journals. 

This paper list will be continuously updated at the end of each month. 


## Table of Contents


- [Survey](#survey)
- [Papers](#papers)
  - [2024](#2024)
  - [2023](#2023)
  - [2022](#2022)
  - [2021](#2021)
  - [2020](#2020)
- [Other Resources](#other-resources)

## Survey

- A Systematic Review of Deep Learning-based Research on Radiology Report Generation (**arXiv 2311**) [[paper](https://arxiv.org/abs/2311.14199)]
- A Survey of Deep Learning-based Radiology Report Generation Using Multimodal Data (**arXiv 2405**) [[paper](https://arxiv.org/abs/2405.12833)]
- Automated Radiology Report Generation: A Review of Recent Advances (**IEEE Reviews in Biomedical Engineering'24**) [[paper](https://ieeexplore.ieee.org/abstract/document/10545538)]

## Dataset
- MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs (**MIMIC-CXR**)  [[paper](https://arxiv.org/abs/1901.07042)][[data](https://physionet.org/content/mimic-cxr/2.0.0/)].
- Preparing a collection of radiology examinations for distribution and retrieval (**IU X-ray**) [[paper](https://academic.oup.com/jamia/article/23/2/304/2572395)][[data](https://openi.nlm.nih.gov/gridquery?q=pneumonia&it=xg&m=1&n=100)].
- Learning Visual-Semantic Embeddings for Reporting Abnormal Findings on Chest X-rays (**MIMIC-ABN**) [[paper](https://aclanthology.org/2020.findings-emnlp.176/)][[code](https://github.com/nijianmo/chest-xray-cvse)]
- An efficient but effective writer: Diffusion-based semi-autoregressive transformer for automated radiology report generation (**XRG-COVID-19**) [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809423010844)][[data](https://github.com/Report-Generation/XRG-COVID-19)].
- HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction (**HistGen WSI**) [[paper](https://arxiv.org/abs/2403.05396)][[data](https://github.com/dddavid4real/HistGen)].

## Metrics
- FineRadScore: A Radiology Report Line-by-Line Evaluation Technique Generating Corrections with Severity Scores (**arXiv'2405**) [[paper](https://arxiv.org/pdf/2405.20613)][[code](https://github.com/rajpurkarlab/FineRadScore)]

## Papers

### 2024

**AAAI'24**
- Automatic Radiology Reports Generation via Memory Alignment Network [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28279)] [code]
- PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28038)][[code]]
- Bootstrapping Large Language Models for Radiology Report Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29826)] [[code](https://github.com/synlp/R2-LLM)]

**CVPR'24**
- ** [[paper]()] [[code]()]

**ACL'24**
- ** [[paper]()] [[code]()]

**EMNLP'24**
- ** [[paper]()] [[code]()]

**MICCAI'24**
- Textual Inversion and Self-supervised Refinement for Radiology Report Generation [[paper](https://arxiv.org/pdf/2405.20607)] [[code]]

**BIBM'24**
- ** [[paper]()] [[code]()]

**ICASSP'24**
- ** [[paper]()] [[code]()]

**MedIA'24**
- ** [[paper]()] [[code]()]

**TMI'24**
- Multi-grained Radiology Report Generation with Sentence-level Image-language Contrastive Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10458706)] [[code]()]

**TCSVT'24**
- ** [[paper]()] [[code]()]

**TNNLS'24**
- ** [[paper]()] [[code]()]

**TMM'24**
- ** [[paper]()] [[code]()]

**JBHI'24**
- CAMANet: Class Activation Map Guided Attention Network for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10400776)] [[code](https://github.com/Markin-Wang/CAMANet)]
- TSGET: Two-Stage Global Enhanced Transformer for Automatic Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10381879)] [[code](https://github.com/Markin-Wang/CAMANet)]
- CAMANet: Class Activation Map Guided Attention Network for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10400776)] [[code](https://github.com/Markin-Wang/CAMANet)]

**arXiv papers'24**
- Factual Serialization Enhancement: A Key Innovation for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2405.09586)] [[code](https://github.com/mk-runner/FSE)]
- FITA: Fine-grained Image-Text Aligner for Radiology Report Generation [[paper](https://arxiv.org/abs/2405.00962)] [[code]]
- GREEN: Generative Radiology Report Evaluation and Error Notation [[paper](https://arxiv.org/abs/2405.03595)] [[code]]
- CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images and Patients [[paper](https://arxiv.org/abs/2405.19538)] [[code]]
- Topicwise Separable Sentence Retrieval for Medical Report Generation [[paper](https://arxiv.org/abs/2405.04175)] [[code]]
- Dia-LLaMA: Towards Large Language Model-driven CT Report Generation [[paper](https://arxiv.org/abs/2403.16386)] [[code]]
- ICON: Improving Inter-Report Consistency of Radiology Report Generation via Lesion-aware Mix-up Augmentation [[paper](https://arxiv.org/abs/2402.12844)] [[code](https://github.com/wjhou/ICon)]
- CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging [[paper](https://arxiv.org/abs/2403.06801)] [[code](https://github.com/ibrahimethemhamamci/CT2Rep)]
- Structural Entities Extraction and Patient Indications Incorporation for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2405.14905)] [[code](https://github.com/mk-runner/SEI-Temp)]
- MAIRA-2: Grounded Radiology Report Generation [[paper](https://arxiv.org/pdf/2406.04449)][[code]]

### 2023

**AAAI'23**
- ** [[paper]()] [[code]()]

**CVPR'23**
- KiUT: Knowledge-injected U-Transformer for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10203622)] [[code]]
- METransformer: Radiology report generation by transformer with multiple learnable expert tokens [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_METransformer_Radiology_Report_Generation_by_Transformer_With_Multiple_Learnable_Expert_CVPR_2023_paper.html)][[code]]
- Dynamic Graph Enhanced Contrastive Learning for Chest X-Ray Report Generation [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Dynamic_Graph_Enhanced_Contrastive_Learning_for_Chest_X-Ray_Report_Generation_CVPR_2023_paper.html)] [[code](https://github.com/mlii0117/DCL)]
- Interactive and Explainable Region-guided Radiology Report Generation [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Tanida_Interactive_and_Explainable_Region-Guided_Radiology_Report_Generation_CVPR_2023_paper.html)][[code](https://github.com/ttanida/rgrg)]

**ACL'23**
- ORGAN: Observation-Guided Radiology Report Generation via Tree Reasoning [[paper](https://arxiv.org/abs/2306.06466)] [[code](https://github.com/wjhou/ORGan)]

**EMNLP'23**
- RECAP: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning [[paper](https://aclanthology.org/2023.findings-emnlp.140/)] [[code](https://github.com/wjhou/Recap)]
- Normal-Abnormal Decoupling Memory for Medical Report Generation [[paper](https://aclanthology.org/2023.findings-emnlp.131/)] [[code](https://github.com/kzzjk/NADM)]
- Style-Aware Radiology Report Generation with RadGraph and Few-Shot Prompting [[paper](https://arxiv.org/abs/2310.17811)] [[code]]


**MICCAI'23**
- ** [[paper]()] [[code]()]

**BIBM'23**
- ** [[paper]()] [[code]()]

**ML4H'23**
- Pragmatic Radiology Report Generation [[paper](https://proceedings.mlr.press/v225/nguyen23a.html)] [[code](https://github.com/ChicagoHAI/llm_radiology)]

**ICASSP'23**
- Improving Radiology Report Generation with D 2-Net: When Diffusion Meets Discriminator [[paper](https://ieeexplore.ieee.org/abstract/document/10448326)] [[code]]
  
**MedIA'23**
- Radiology report generation with a learned knowledge base and multi-modal alignment [[paper](https://www.sciencedirect.com/science/article/pii/S1361841523000592)] [[code](https://github.com/LX-doctorAI1/M2KT)]


**TMI'23**
- ** [[paper]()] [[code]()]

**TCSVT'23**
- ** [[paper]()] [[code]()]

**TNNLS'23**
- ** [[paper]()] [[code]()]

**TMM'23**
- From Observation to Concept: A Flexible Multi-view Paradigm for Medical Report Generation [[paper](https://ieeexplore.ieee.org/abstract/document/10356722)] [[code]]

**JBHI'23**
- ** [[paper]()] [[code]()]

**arXiv papers'23**
- MAIRA-1: A specialised large multimodal model for radiology report generation [[paper](https://arxiv.org/abs/2311.13668)] [[code]]

### 2022

**AAAI'22**
- Clinical-BERT: Vision-Language Pre-training for Radiograph Diagnosis and Reports Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20204)] [[code]]

**CVPR'22**
- ** [[paper]()] [[code]()]

**ACL'22**
- Reinforced Cross-modal Alignment for Radiology Report Generation [[paper](https://aclanthology.org/2022.findings-acl.38/)] [[code](https://
github.com/cuhksz-nlp/R2GenRL)]

**EMNLP'22**
- ** [[paper]()] [[code]()]

**MICCAI'22**
- A Medical Semantic-Assisted Transformer for Radiographic Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_63)] [[code](https://github.com/wang-zhanyu/MSAT)]

**BIBM'22**
- ** [[paper]()] [[code]()]

**ICASSP'22**
- ** [[paper]()] [[code]()]

**MedIA'22**
- Knowledge matters: Chest radiology report generation with general and specific knowledge [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001578)] [[code](https://github.com/LX-doctorAI1/GSKET)]

**TMI'22**
- Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach [[paper](https://ieeexplore.ieee.org/document/9768661)] [[code]]

**TCSVT'22**
- ** [[paper]()] [[code]()]

**TNNLS'22**
- ** [[paper]()] [[code]()]
  
**TMM'22**
- ** [[paper]()] [[code]()]

**JBHI'22**
- ** [[paper]()] [[code]()]


**arXiv papers'22**
- ** [[paper]()] [[code]()]

### 2021

**AAAI'21**
- ** [[paper]()] [[code]()]

**CVPR'21**
- ** [[paper]()] [[code]()]

**ACL'21**
- Cross-modal Memory Networks for Radiology Report Generation [[paper](https://aclanthology.org/2021.acl-long.459/)] [[code](https://github.com/zhjohnchan/R2GenCMN)]

**EMNLP'21**
- Progressive Transformer-Based Generation of Radiology Reports [[paper](https://arxiv.org/pdf/2102.09777)] [[code](https://github.com/uzh-dqbm-cmi/ARGON)]
  
**NAACL'21**
- Improving Factual Completeness and Consistency of Image-to-Text Radiology Report Generation [[paper](https://aclanthology.org/2021.naacl-main.416/)] [[code](https://github.com/ysmiura/ifcc)]

**MICCAI'21**
- ** [[paper]()] [[code]()]

**BIBM'21**
- ** [[paper]()] [[code]()]

**ICASSP'21**
- ** [[paper]()] [[code]()]

**MedIA'21**
- ** [[paper]()] [[code]()]

**TMI'21**
- ** [[paper]()] [[code]()]

**TCSVT'21**
- ** [[paper]()] [[code]()]

**TNNLS'21**
- ** [[paper]()] [[code]()]

**TMM'21**
- ** [[paper]()] [[code]()]

**JBHI'21**
- ** [[paper]()] [[code]()]


**arXiv papers'21**
- ** [[paper]()] [[code]()]

### 2020

**AAAI'20**
- When Radiology Report Generation Meets Knowledge Graph [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6989)] [[code]]

**CVPR'20**
- ** [[paper]()] [[code]()]

**ACL'20**
- ** [[paper]()] [[code]()]

**EMNLP'20**
- Generating Radiology Reports via Memory-driven Transformer [[paper](https://arxiv.org/abs/2010.16056)] [[code](https://github.com/zhjohnchan/R2Gen)]

**MICCAI'20**
- ** [[paper]()] [[code]()]

**BIBM'20**
- ** [[paper]()] [[code]()]

**ICASSP'20**
- ** [[paper]()] [[code]()]
  
**MedIA'20**
- ** [[paper]()] [[code]()]

**TMI'20**
- ** [[paper]()] [[code]()]

**TCSVT'20**
- ** [[paper]()] [[code]()]

**TNNLS'20**
- ** [[paper]()] [[code]()]

**TMM'20**
- ** [[paper]()] [[code]()]

**JBHI'20**
- ** [[paper]()] [[code]()]


**arXiv papers'20**
- ** [[paper]()] [[code]()]



## Other Resources
With all the resources available on the Github website, this paper list is comprehensive and recently updated.

- [**]()

## Tools
- [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)
- [coco-caption](https://github.com/tylin/coco-caption)
- [f1chexbert](https://pypi.org/project/f1chexbert/)
- [radgraph](https://pypi.org/project/radgraph/)


## Last update: Jun 7, 2024

## Feel free to contact me if you find any interesting papers are missing.
email: kangliu422@gmail.com
