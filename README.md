# Awesome-Radiology-Report-Generation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

We collect existing papers on radiology report generation published in prominent conferences and journals. 

This paper list will be continuously updated at the end of each month. 


## Table of Contents


- [Survey](#survey)
- [Dataset](#Dataset)
- [Metrics](#Metrics)
- [Foundation Models for Medicine](#Foundation-Models-for-Medicine)
- [Papers](#papers)
  - [2024](#2024)
  - [2023](#2023)
  - [2022](#2022)
  - [2021](#2021)
  - [2020](#2020)
- [Other Resources](#other-resources)
- [Tools](#Tools)

## Survey

- A Systematic Review of Deep Learning-based Research on Radiology Report Generation (**arXiv 2311**) [[paper](https://arxiv.org/abs/2311.14199)]
- A Survey of Deep Learning-based Radiology Report Generation Using Multimodal Data (**arXiv 2405**) [[paper](https://arxiv.org/abs/2405.12833)]
- Automated Radiology Report Generation: A Review of Recent Advances (**IEEE Reviews in Biomedical Engineering'24**) [[paper](https://ieeexplore.ieee.org/abstract/document/10545538)]
- From Vision to Text: A Comprehensive Review of Natural Image Captioning in Medical Diagnosis and Radiology Report Generation (**Medical Image Analysis**)[[paper](https://www.sciencedirect.com/science/article/pii/S1361841524001890)]

## Dataset
- MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs (**MIMIC-CXR**)  [[paper](https://arxiv.org/abs/1901.07042)][[data](https://physionet.org/content/mimic-cxr/2.0.0/)].
- Preparing a collection of radiology examinations for distribution and retrieval (**IU X-ray**) [[paper](https://academic.oup.com/jamia/article/23/2/304/2572395)][[data](https://openi.nlm.nih.gov/gridquery?q=pneumonia&it=xg&m=1&n=100)].
- Learning Visual-Semantic Embeddings for Reporting Abnormal Findings on Chest X-rays (**MIMIC-ABN**) [[paper](https://aclanthology.org/2020.findings-emnlp.176/)][[code](https://github.com/nijianmo/chest-xray-cvse)]
- An efficient but effective writer: Diffusion-based semi-autoregressive transformer for automated radiology report generation (**XRG-COVID-19**) [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809423010844)][[data](https://github.com/Report-Generation/XRG-COVID-19)].
- HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction (**HistGen WSI**) [[paper](https://arxiv.org/abs/2403.05396)][[data](https://github.com/dddavid4real/HistGen)].
- CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images and Patients (**CheXpert Plus**) [[paper](https://arxiv.org/abs/2405.19538)] [[data](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1)]
- CXR-PRO: MIMIC-CXR with Prior References Omitted (**CXR-PRO**) [[data](https://www.physionet.org/content/cxr-pro/1.0.0/)]
- MS-CXR: Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing (**MS-CXR**) [[data](https://physionet.org/content/ms-cxr/0.1/)]
- EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images (**EHRXQA**)[[paper](https://neurips.cc/virtual/2023/poster/73600)][[code](https://github.com/baeseongsu/ehrxqa)][[data](https://physionet.org/content/ehrxqa/1.0.0/)]
- MIMIC-Ext-MIMIC-CXR-VQA: A Complex, Diverse, And Large-Scale Visual Question Answering Dataset for Chest X-ray Images (**MIMIC-Ext-MIMIC-CXR-VQA**)[[code](https://github.com/baeseongsu/mimic-cxr-vqa)][[data](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/MIMIC-Ext-MIMIC-CXR-VQA/)]
- MS-CXR-T: Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing (**MS-CXR-T**)[[data](https://physionet.org/content/ms-cxr-t/1.0.0/)]

## Metrics
- FineRadScore: A Radiology Report Line-by-Line Evaluation Technique Generating Corrections with Severity Scores (**arXiv'2405**) [[paper](https://arxiv.org/pdf/2405.20613)][[code](https://github.com/rajpurkarlab/FineRadScore)]
- FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation (**EMNLP'23**) [[paper](https://arxiv.org/abs/2305.14251)][[code](https://github.com/shmsw25/FActScore?tab=readme-ov-file)]
- DocLens: Multi-aspect Fine-grained Evaluation for Medical Text Generation (**ACL'24**) [[paper](https://arxiv.org/abs/2311.09581)][[code](https://github.com/yiqingxyq/DocLens)]
- RaTEScore: A Metric for Radiology Report Generation [[paper](https://www.medrxiv.org/content/10.1101/2024.06.24.24309405v1)][[code](https://github.com/MAGIC-AI4Med/RaTEScore)]
- GREEN: Generative Radiology Report Evaluation and Error Notation [[paper](https://arxiv.org/pdf/2405.03595)][[code](https://github.com/Stanford-AIMI/GREEN)]
- When Radiology Report Generation Meets Knowledge Graph (**MIRQI**) [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6989)][[code](https://github.com/xiaosongwang/MIRQI)]
- Evaluating progress in automatic chest X-ray radiology report generation (**RadCliQ**)[[paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5)][[code](https://github.com/rajpurkarlab/CXR-Report-Metric)]
- Evaluating GPT-4 on Impressions Generation in Radiology Reports (**Radiology**)[[paper](https://pubs.rsna.org/doi/full/10.1148/radiol.231259)]

## Foundation Models for Medicine
- CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation (**arXiv'2401**) [[paper](https://stanford-aimi.github.io/chexagent.html)][[code](https://github.com/Stanford-AIMI/CheXagent)]

## Papers

### 2024

**AAAI'24**
- Automatic Radiology Reports Generation via Memory Alignment Network [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28279)] [code]
- PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28038)][[code](https://github.com/synlp/R2-LLM)]
- Bootstrapping Large Language Models for Radiology Report Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29826)] [[code](https://github.com/synlp/R2-LLM)]

**CVPR'24**
- Instance-level Expert Knowledge and Aggregate Discriminative Attention for Radiology Report Generation [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Bu_Instance-level_Expert_Knowledge_and_Aggregate_Discriminative_Attention_for_Radiology_Report_CVPR_2024_paper.html)] [[code](https://github.com/hnjzbss/EKAGen)]
- AHIVE: Anatomy-aware Hierarchical Vision Encoding for Interactive Radiology Report Retrieval [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_AHIVE_Anatomy-aware_Hierarchical_Vision_Encoding_for_Interactive_Radiology_Report_Retrieval_CVPR_2024_paper.html)] [[code]]
- InVERGe: Intelligent Visual Encoder for Bridging Modalities in Report Generation [[paper](https://openaccess.thecvf.com/content/CVPR2024W/MULA/papers/Deria_InVERGe_Intelligent_Visual_Encoder_for_Bridging_Modalities_in_Report_Generation_CVPRW_2024_paper.pdf)][[code]( https://github.com/labsroy007/InVERGe)]

**ACL'24**
- DocLens: Multi-aspect Fine-grained Evaluation for Medical Text Generation [[paper](https://arxiv.org/abs/2311.09581)][[code](https://github.com/yiqingxyq/DocLens)]

**ACM MM'24**
- Medical Report Generation via Multimodal Spatio-Temporal Fusion [[paper](https://openreview.net/pdf?id=XKs7DR9GAK)]
- Diffusion Networks with Task-Specific Noise Control for Radiology Report Generation [[paper](https://openreview.net/pdf?id=kbdeQmw2ny)]
- Divide and Conquer: Isolating Normal-Abnormal Attributes in Knowledge Graph-Enhanced Radiology Report Generation [[paper](https://openreview.net/forum?id=TuU8TQVOoj)]
- In-context Learning for Zero-shot Medical Report Generation [[paper](https://openreview.net/pdf?id=8zyG2eUgVE)]

**ECCV'24**
- HERGen: Elevating Radiology Report Generation with Longitudinal Data [[paper](https://arxiv.org/pdf/2407.15158)] [[code](https://github.com/fuying-wang/HERGen)]
- Contrastive Learning with Counterfactual Explanations for Radiology Report Generation [[paper](https://arxiv.org/abs/2407.14474)]

**EMNLP'24**
- ** [[paper]()] [[code]()]

**MICCAI'24**
- Textual Inversion and Self-supervised Refinement for Radiology Report Generation [[paper](https://arxiv.org/pdf/2405.20607)] [[code]]
- Structural Entities Extraction and Patient Indications Incorporation for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2405.14905)] [[code](https://github.com/mk-runner/SEI)]
- CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging [[paper](https://arxiv.org/abs/2403.06801)] [[code](https://github.com/ibrahimethemhamamci/CT2Rep)]
- WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole Slide Images [[paper](https://arxiv.org/abs/2311.16480)][[code](https://github.com/cpystan/Wsi-Caption)]

**BIBM'24**
- ** [[paper]()] [[code]()]

**MedIA'24**
- From Vision to Text: A Comprehensive Review of Natural Image Captioning in Medical Diagnosis and Radiology Report Generation [[paper](https://www.sciencedirect.com/science/article/pii/S1361841524001890)]

**TMI'24**
- Multi-grained Radiology Report Generation with Sentence-level Image-language Contrastive Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10458706)] [[code]]
- SGT++: Improved Scene Graph-Guided Transformer for Surgical Report Generation [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10330637)][[code]]
- PhraseAug: An Augmented Medical Report Generation Model with Phrasebook [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10560051)] [[code]]
- Token-Mixer: Bind Image and Text in One Embedding Space for Medical Image Reporting [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10552817)] [[code](https://github.com/yangyan22/Token-Mixer)]
- An Organ-aware Diagnosis Framework for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10579857)]
- Attribute Prototype-guided Iterative Scene Graph for Explainable Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10587279)]

**TMM'24**
- Semi-Supervised Medical Report Generation via Graph-Guided Hybrid Feature Consistency [[paper](https://ieeexplore.ieee.org/document/10119200)] [[code]]
- Multi-Level Objective Alignment Transformer for Fine-Grained Oral Panoramic X-Ray Report Generation [[paper](https://ieeexplore.ieee.org/document/10443573)] [[code]]

**JBHI'24**
- CAMANet: Class Activation Map Guided Attention Network for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10400776)] [[code](https://github.com/Markin-Wang/CAMANet)]
- TSGET: Two-Stage Global Enhanced Transformer for Automatic Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10381879)] [[code](https://github.com/Markin-Wang/CAMANet)]

**Expert Systems with Applications'24**
- CheXReport: A transformer-based architecture to generate chest X-ray reports suggestions [[paper](https://www.sciencedirect.com/science/article/pii/S0957417424015112)][[code](https://github.com/felipezeiser/CheXReport)]

**Neurocomputing'24**
- Improving radiology report generation with multi-grained abnormality prediction [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224008932)]

**arXiv papers'24**
- Factual Serialization Enhancement: A Key Innovation for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2405.09586)] [[code](https://github.com/mk-runner/FSE)]
- FITA: Fine-grained Image-Text Aligner for Radiology Report Generation [[paper](https://arxiv.org/abs/2405.00962)] [[code]]
- GREEN: Generative Radiology Report Evaluation and Error Notation [[paper](https://arxiv.org/abs/2405.03595)] [[code]]
- CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images and Patients [[paper](https://arxiv.org/abs/2405.19538)] [[code](https://github.com/Stanford-AIMI/chexpert-plus)]
- Topicwise Separable Sentence Retrieval for Medical Report Generation [[paper](https://arxiv.org/abs/2405.04175)] [[code]]
- Dia-LLaMA: Towards Large Language Model-driven CT Report Generation [[paper](https://arxiv.org/abs/2403.16386)] [[code]]
- ICON: Improving Inter-Report Consistency of Radiology Report Generation via Lesion-aware Mix-up Augmentation [[paper](https://arxiv.org/abs/2402.12844)] [[code](https://github.com/wjhou/ICon)]
- MAIRA-2: Grounded Radiology Report Generation [[paper](https://arxiv.org/pdf/2406.04449)][[code]]
- Benchmarking and Boosting Radiology Report Generation for 3D High-Resolution Medical Images [[paper](https://arxiv.org/pdf/2406.07146)]
- The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It [[paper](https://arxiv.org/pdf/2406.13181)][[code](https://anonymous.4open.science/r/anon-D83E/README.md)]
- Improving Expert Radiology Report Summarization by Prompting Large Language Models with a Layperson Summary [[paper](https://arxiv.org/pdf/2406.13181)]
- Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation [[paper](https://arxiv.org/pdf/2407.15268)]
- X-ray Made Simple: Radiology Report Generation and Evaluation with Layman's Terms [[paper](https://arxiv.org/abs/2406.17911)]
- Multi-modal vision-language model for generalizable annotation-free pathology localization and clinical diagnosis [[paper](https://arxiv.org/pdf/2401.02044)][[code](https://github.com/YH0517/AFLoc)]
- Direct Preference Optimization for Suppressing Hallucinated Prior Exams in Radiology Report Generation [[paper]](https://arxiv.org/pdf/2406.06496)]


### 2023
**ICLR'23**
- Advancing radiograph representation learning with masked record modeling [[paper](https://openreview.net/forum?id=w-x7U26GM7j)][[code](https://github.com/RL4M)]

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
- PhenotypeCLIP: Phenotype-based Contrastive Learning for Medical Imaging Report Generation [[paper](https://aclanthology.org/2023.emnlp-main.989.pdf)]


**MICCAI'23**
- Utilizing Longitudinal Chest X-Rays and Reports to Pre-Fill Radiology Reports [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_19)] [[code](https://github.com/CelestialShine/Longitudinal-Chest-X-Ray)]

**BIBM'23**
- ** [[paper]()] [[code]()]

**ML4H'23**
- Pragmatic Radiology Report Generation [[paper](https://proceedings.mlr.press/v225/nguyen23a.html)] [[code](https://github.com/ChicagoHAI/llm_radiology)]

**ICASSP'23**
- Improving Radiology Report Generation with D 2-Net: When Diffusion Meets Discriminator [[paper](https://ieeexplore.ieee.org/abstract/document/10448326)] [[code]]
  
**MedIA'23**
- Radiology report generation with a learned knowledge base and multi-modal alignment [[paper](https://www.sciencedirect.com/science/article/pii/S1361841523000592)] [[code](https://github.com/LX-doctorAI1/M2KT)]

**TMI'23**
- Attributed Abnormality Graph Embedding for Clinically Accurate X-Ray Report Generation [[paper](https://ieeexplore.ieee.org/document/10045710)][[code]]

**Patterns'23**
- Evaluating progress in automatic chest X-ray radiology report generation[[paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5)][[code](https://github.com/rajpurkarlab/CXR-Report-Metric)]

**TMM'23**
- From Observation to Concept: A Flexible Multi-view Paradigm for Medical Report Generation [[paper](https://ieeexplore.ieee.org/abstract/document/10356722)] [[code]]
- Joint Embedding of Deep Visual and Semantic Features for Medical Image Report Generation [[paper](https://ieeexplore.ieee.org/document/9606584)] [[code]]


**arXiv papers'23**
- MAIRA-1: A specialised large multimodal model for radiology report generation [[paper](https://arxiv.org/abs/2311.13668)] [[code]]
- Longitudinal Data and a Semantic Similarity Reward for Chest X-Ray Report Generation [[paper](https://arxiv.org/abs/2307.09758)][[code](https://github.com/aehrc/cxrmate)]

### 2022

**AAAI'22**
- Clinical-BERT: Vision-Language Pre-training for Radiograph Diagnosis and Reports Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20204)] [[code]]


**ACL'22**
- Reinforced Cross-modal Alignment for Radiology Report Generation [[paper](https://aclanthology.org/2022.findings-acl.38/)] [[code](https://github.com/cuhksz-nlp/R2GenRL)]


**MICCAI'22**
- A Medical Semantic-Assisted Transformer for Radiographic Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_63)] [[code](https://github.com/wang-zhanyu/MSAT)]
- CheXRelNet An Anatomy-Aware Model for Tracking Longitudinal Relationships Between Chest X-Rays [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_55)][[code](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_55)]

**Nature Machine Intelligence'22**
- Generalized radiograph representation learning via cross-supervision between images and free-text radiology reports [[paper](https://www.nature.com/articles/s42256-021-00425-9)][[code](https://github.com/funnyzhou/REFERS)]

**MedIA'22**
- Knowledge matters: Chest radiology report generation with general and specific knowledge [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001578)] [[code](https://github.com/LX-doctorAI1/GSKET)]

**TMI'22**
- Automated Radiographic Report Generation Purely on Transformer: A Multicriteria Supervised Approach [[paper](https://ieeexplore.ieee.org/document/9768661)] [[code]]

**arXiv papers'22**
- ** [[paper]()] [[code]()]

### 2021

**ACL'21**
- Cross-modal Memory Networks for Radiology Report Generation [[paper](https://aclanthology.org/2021.acl-long.459/)] [[code](https://github.com/zhjohnchan/R2GenCMN)]

**EMNLP'21**
- Progressive Transformer-Based Generation of Radiology Reports [[paper](https://arxiv.org/pdf/2102.09777)] [[code](https://github.com/uzh-dqbm-cmi/ARGON)]
  
**NAACL'21**
- Improving Factual Completeness and Consistency of Image-to-Text Radiology Report Generation [[paper](https://aclanthology.org/2021.naacl-main.416/)] [[code](https://github.com/ysmiura/ifcc)]

### 2020

**AAAI'20**
- When Radiology Report Generation Meets Knowledge Graph [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6989)] [[code](https://github.com/xiaosongwang/MIRQI)]

**EMNLP'20**
- Generating Radiology Reports via Memory-driven Transformer [[paper](https://arxiv.org/abs/2010.16056)] [[code](https://github.com/zhjohnchan/R2Gen)]


## Other Resources
- Learning to Exploit Temporal Structure for Biomedical Vision–Language Processing (**CVPR'23**) [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.pdf)
[[code](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)]

## Tools
- [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)
- [coco-caption](https://github.com/tylin/coco-caption)
- [f1chexbert](https://pypi.org/project/f1chexbert/)
- [radgraph](https://pypi.org/project/radgraph/)
- [mimic-cxr](https://github.com/MIT-LCP/mimic-cxr)


## Feel free to contact me if you find any interesting papers missing.
email: kangliu422@gmail.com
