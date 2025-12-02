# Awesome-Radiology-Report-Generation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

We collect existing papers on radiology report generation that have been published in prominent conferences and journals. If you find this helpful, please consider citing the following reference.
```
@InProceedings{Liu-2025-CVPR,
    author    = {Liu, Kang and Ma, Zhuoqi and Kang, Xiaolu and Li, Yunan and Xie, Kun and Jiao, Zhicheng and Miao, Qiguang},
    title     = {Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {10348-10359}
}

@misc{2026-AAAI-priorrg,
  title={PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation},
  author={Kang Liu and Zhuoqi Ma and Zikang Fang and Yunan Li and Kun Xie and Qiguang Miao},
  year={2025},
  eprint={2508.05353},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.05353}
}
```

## Table of Contents

- [Foundation Models for Medicine](#Foundation-Models-for-Medicine)
- [Papers](#papers)
  - [2026](#2026)
  - [2025](#2025)
  - [2024](#2024)
  - [2023](#2023)
  - [2022](#2022)
  - [2021](#2021)
  - [2020](#2020)
- [Survey](#survey)
- [Dataset](#Dataset)
- [Metrics](#Metrics)
- [Other Resources](#other-resources)
- [Tools](#Tools)


## Foundation Models for Medicine
- CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation (**arXiv'2401**) [[paper](https://stanford-aimi.github.io/chexagent.html)][[code](https://github.com/Stanford-AIMI/CheXagent)]
- XrayGPT: Chest Radiographs Summarization using Large Medical Vision-Language Models (**ACLW'24**)[[paper](https://aclanthology.org/2024.bionlp-1.35.pdf)][[code](https://github.com/mbzuai-oryx/XrayGPT)]
- Unlocking the Power of Spatial and Temporal Information in Medical Multimodal Pre-training (**ICML'24**) [[paper](https://arxiv.org/pdf/2405.19654)][[code](https://github.com/SVT-Yang/MedST)]
- A generalist vision--language foundation model for diverse biomedical tasks (**Nature Medicine'24**)[[paper](https://www.nature.com/articles/s41591-024-03185-2)][[code](https://github.com/taokz/BiomedGPT)]
- ECAMP: Entity-centered Context-aware Medical Vision Language Pre-training (**arXiv'2311**)[[paper](https://arxiv.org/pdf/2312.13316)][[code](https://github.com/ToniChopp/ECAMP)]
- CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training (**MICCAI'23**)[[paper](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_10)][[code](https://github.com/kakaobrain/cxr-clip)]
- GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition (**ICCV'21**)[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf)][[code](https://github.com/marshuang80/gloria)]
- CXR-LLAVA: a multimodal large language model for interpreting chest X-ray images (**arXiv'2310**)[[paper](https://arxiv.org/pdf/2310.18341)][[code](https://github.com/ECOFRI/CXR_LLAVA)]
- LLaVA-OneVision: Easy Visual Task Transfer (**arXiv'2408**)[[paper](https://arxiv.org/pdf/2408.03326)][[code](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)]
- Advancing Medical Radiograph Representation Learning: A Hybrid Pre-training Paradigm with Multilevel Semantic Granularity (**arXiv'2410**) [[paper](https://arxiv.org/pdf/2410.00448)]
- MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging (**arXiv'2410**)[[paper](https://arxiv.org/abs/2410.06542)]
- BioBridge: Bridging Biomedical Foundation Models via Knowledge Graphs (**ICLR'24**)[[paper](https://arxiv.org/pdf/2310.03320)][[code](https://github.com/RyanWangZf/BioBridge)]
- Eye-gaze Guided Multi-modal Alignment for Medical Representation Learning (**NIPS'24**) [[paper](https://openreview.net/pdf?id=0bINeW40u4)]
- Advancing human-centric AI for robust X-ray analysis through holistic self-supervised learning (**arXiv'2405**)[[paper](https://arxiv.org/pdf/2405.01469)]
- MAIRA-2: Grounded Radiology Report Generation (**arXiv'2406**)[[paper](https://arxiv.org/abs/2406.04449)]
- A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings (**Nature Communications'25**)[[paper](https://www.nature.com/articles/s41467-025-58344-x)]



## Papers
### 2026
**AAAI'26**
- PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2508.05353)][[code](https://github.com/mk-runner/PriorRG)]
- S2D-ALIGN: Shallow-to-Deep Auxiliary Learning for Anatomically-Grounded Radiology Report Generation [[paper](https://arxiv.org/pdf/2511.11066)]
- A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation [[paper](https://arxiv.org/pdf/2511.12259)]

**EAAI'26**
- Auto-encoding clinical language for zero-shot medical report generation [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625032750)]

---

### 2025

**Nature'25**
- A fully open AI foundation model applied to chest radiography [[paper](https://www.nature.com/articles/s41586-025-09079-8)]

**Nature Medicine'25**
- A generalist medical language model for disease diagnosis assistance [[paper](https://www.nature.com/articles/s41591-024-03416-6)][[code](https://github.com/medfound/medfound)]

**Nature Communications'25**
- Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation [[paper](https://www.nature.com/articles/s41467-025-57426-0)]
- A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings [[paper](https://www.nature.com/articles/s41467-025-58344-x)][[code](https://github.com/microsoft/llava-rad)][[metric](https://github.com/microsoft/chexprompt)]
- Generating dermatopathology reports from gigapixel whole slide images with HistoGPT [[paper](https://www.nature.com/articles/s41467-025-60014-x)]

**TPAMI'25**
- Diagnostic Captioning by Cooperative Task Interactions and Sample-graph Consistency [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10974505)]
- Uncertainty-aware Medical Diagnostic Phrase Identification and Grounding [[paper](https://arxiv.org/pdf/2404.06798)][[code](https://github.com/Cocofeat/uMedGround)]

**Nature Computational Science'25**
- Evaluating and Mitigating Bias in AI-Based Medical Text Generation [[paper](https://arxiv.org/pdf/2504.17279v1)] [[code](https://github.com/iriscxy/GenFair)]
- Toward fair AI-driven medical text generation [[paper](https://www.nature.com/articles/s43588-025-00807-8)]

**npj Digital Medicine'25**
- Keyword-based AI assistance in the generation of radiology reports: A pilot study [[paper](https://www.nature.com/articles/s41746-025-01889-4)]
- A multimodal multidomain multilingual medical foundation model for zero shot clinical diagnosis [[paper](https://www.nature.com/articles/s41746-024-01339-7)][[code](https://github.com/AI-in-Health/M3FM)]
- A deep learning based automatic report generator for retinal optical coherence tomography images [[paper](https://www.nature.com/articles/s41746-025-01988-2)]

**NEJM AI'25**
- PadChest-GR: A Bilingual Chest X-Ray Dataset for Grounded Radiology Report Generation [[paper](https://ai.nejm.org/doi/full/10.1056/AIdbp2401120)]

**Radiology: Artificial Intelligence'25**
- Retrieval-Augmented Generation with Large Language Models in Radiology: From Theory to Practice [[paper](https://pubs.rsna.org/doi/abs/10.1148/ryai.240790)]

**JAMA Network Open'25**
- Efficiency and Quality of Generative AI–Assisted Radiograph Reporting [[paper](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2834943)]

**CVPR'25**
- Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2502.20056)] [[code](https://github.com/mk-runner/MLRG)]
- FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models [[paper](https://arxiv.org/pdf/2411.18672)]
- DART: Disease-aware Image-Text Alignment and Self-correcting Re-alignment for Trustworthy Radiology Report Generation [[paper](https://cvpr.thecvf.com/virtual/2025/poster/32986)]
- CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset [[paper](https://arxiv.org/pdf/2410.00379)][[code](https://github.com/Event-AHU/Medical_Image_Analysis)]
- VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge [[paper](https://arxiv.org/pdf/2411.12915)][[code](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework)]
- Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation [[paper](https://arxiv.org/pdf/2504.18856v1)][[code](https://github.com/BasitAlawode/MR-PLIP)]
- ARDGen: Augmentation Regularization for Domain-Generalized Medical Report Generation (**Workshop**) [[paper](https://openaccess.thecvf.com/content/CVPR2025W/DG-EBF/papers/Ahsan_ARDGen_Augmentation_Regularization_for_Domain-Generalized_Medical_Report_Generation_CVPRW_2025_paper.pdf)]
- MVCM: Enhancing Multi-View and Cross-Modality Alignment for Medical Visual Question Answering and Medical Image-Text Retrieval (**Workshop**) [[paper](https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/papers/Zou_MVCM_Enhancing_Multi-View_and_Cross-Modality_Alignment_for_Medical_Visual_Question_CVPRW_2025_paper.pdf)]
- Alignment, Mining and Fusion: Representation Alignment with Hard Negative Mining and Selective Knowledge Fusion for Medical Visual Question Answering [[paper](https://arxiv.org/pdf/2510.08791v1)][[code](https://github.com/AlexCo1d/AMiF)]

**ICLR'25**
- MedRegA: Interpretable Bilingual Multimodal Large Language Model for Diverse Biomedical Tasks [[paper](https://arxiv.org/abs/2410.18387)][[code](https://github.com/xmed-lab/MedRegA)]

**ICCV'25**
- GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis (**GEMeX**)[[paper](https://arxiv.org/pdf/2411.16778)][[project](https://huggingface.co/datasets/BoKelvin/GEMeX)]
- CT-GRAPH: Hierarchical Graph Attention Network for Anatomy-Guided CT Report Generation (**workshop**) [[paper](https://arxiv.org/pdf/2508.05375?)][[code](https://github.com/hakal104/CT-GRAPH)]
- Knowledge-Driven Query Network with Adaptive Cross-View Attention for Structured Radiology Report Generation (**workshop**)[[paper](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Hou_Knowledge-Driven_Query_Network_with_Adaptive_Cross-View_Attention_for_Structured_Radiology_ICCVW_2025_paper.pdf)]
- RadGPT: Constructing 3D Image-Text Tumor Datasets (**RadGPT**)[[paper](https://www.cs.jhu.edu/~zongwei/publication/bassi2025radgpt.pdf)][[project](https://www.zongweiz.com/dataset)]

**NIPS'25**
- Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation[[paper](https://arxiv.org/pdf/2509.24739v1)]
- CURV: Coherent Uncertainty-Aware Reasoning in Vision-Language Models for X-Ray Report Generation [[paper](https://neurips.cc/virtual/2025/poster/120063)][[code](https://github.com/wwwadx/CURV)]
- Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment [[paper](https://arxiv.org/pdf/2510.11112)]
- Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation [[paper](https://arxiv.org/abs/2509.24739)][[code](https://github.com/AIoT-Lab-AI4LIFE/ViPET-ReportGen)]

**AAAI'25**
- Radiology Report Generation via Multi-objective Preference Optimization [[paper](https://arxiv.org/pdf/2412.08901)]
- HC-LLM: Historical-Constrained Large Language Models for Radiology Report Generation [[paper](https://www.arxiv.org/pdf/2412.11070)][[code](https://github.com/TengfeiLiu966/HC-LLM)]
- LLM-RG4: Flexible and Factual Radiology Report Generation across Diverse Input Contexts [[paper](https://arxiv.org/pdf/2412.12001)][[code](https://github.com/zh-Wang-Med/LLM-RG4)]
- MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation [[paper](https://arxiv.org/pdf/2503.17784)][[code](https://github.com/YanzhaoShi/MEPNet)]
- Overcoming Heterogeneous Data in Federated Medical Vision-Language Pre-training: A Triple-Embedding Model Selector Approach [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/32807)][[code](https://github.com/NBT-AILAB/PMS-FM)]
- DAMPER: A Dual-Stage Medical Report Generation Framework with Coarse-Grained MeSH Alignment and Fine-Grained Hypergraph Matching [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/32393)]

**ICML'25**
- MedRAX: Medical Reasoning Agent for Chest X-ray [[paper](https://arxiv.org/pdf/2502.02673)][[code](https://github.com/bowang-lab/MedRAX)]

**ACL'25**
- RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection [[paper](https://arxiv.org/pdf/2505.14318v1)][[code](https://github.com/wjhou/Radar)]
- Libra: Leveraging Temporal Images for Biomedical Radiology Analysis [[paper](https://arxiv.org/abs/2411.19378)][[code](https://github.com/X-iZhang/Libra)]
- Online Iterative Self-Alignment for Radiology Report Generation [[paper](https://aclanthology.org/2025.acl-long.1348.pdf)]
- Automated Structured Radiology Report Generation [[paper](https://aclanthology.org/2025.acl-long.1301/)]
- The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It [[paper](https://aclanthology.org/2025.acl-long.9.pdf)]
- MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation [[paper](https://arxiv.org/abs/2403.04945)][[code](https://github.com/AIoT-MLSys-Lab/MEIT)]
- CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization [[paper](https://arxiv.org/pdf/2503.05750)]
- Look & Mark: Leveraging Radiologist Eye Fixations and Bounding boxes in Multimodal Large Language Models for Chest X-ray Report Generation [[paper](https://arxiv.org/pdf/2505.22222)]
- Argus: Benchmarking and Enhancing Vision-Language Models for 3D Radiology Report Generation [[paper](https://aclanthology.org/2025.findings-acl.845.pdf)]

**ACMMM'25**
- CheXPO: Preference Optimization for Chest X-ray VLMs with Counterfactual Rationale [[paper](https://arxiv.org/pdf/2507.06959)][[code](https://github.com/ResearchGroup-MedVLLM/CheX-Phi35V)]
- Self-Supervised Anatomical Consistency Learning for Vision-Grounded Medical Report Generation [[paper](https://arxiv.org/pdf/2509.25963)]
- Cross-Counter-Repeat Attention for Enhanced Understanding of Visual Semantics in Radiology Report Generation [[paper](https://dl.acm.org/doi/abs/10.1145/3746027.3755368)]
- Pathology-Aware Reconstruction with Discriminative Knowledge Boosting Alignment for Che-Xray Vision-Language Pre-training [[paper](https://dl.acm.org/doi/abs/10.1145/3746027.3755336)]
- Medical Vision-Language Pre-training with Multimodal Variational Masked Autoencoder for Robust Medical VQA [[paper](https://dl.acm.org/doi/abs/10.1145/3746027.3755273)]

**IJCAI'25**
- RRG-Mamba: Efficient Radiology Report Generation with State Space Model [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4573.pdf)][[code](https://github.com/Eleanorhxd/RRG-Mamba)]
- Cyclic Vision-Language Manipulator: Towards Reliable and Fine-Grained Image Interpretation for Automated Report Generation [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1579.pdf)]

**EMNLP'25**
- CREPE: Rapid Chest X-ray Report Evaluation by Predicting Multi-category Error Counts [[paper](https://openreview.net/forum?id=gjFuz5jbiJ)][[code](https://github.com/gihuncho/crepe/tree/main)]
- MediVLM: A Vision Language Model for Radiology Report Generation from Medical Images [[paper](https://aclanthology.org/2025.findings-emnlp.544.pdf)]
- MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education [[paper](https://aclanthology.org/2025.emnlp-demos.24.pdf)]
- The More, The Better? A Critical Study of Multimodal Context in Radiology Report Summarization [[paper](https://aclanthology.org/2025.findings-emnlp.1040.pdf)]

**COLING'25**
- KIA: Knowledge-Guided Implicit Vision-Language Alignment for Chest X-Ray Report Generation [[paper](https://aclanthology.org/2025.coling-main.276/)]
- CmEAA: Cross-modal Enhancement and Alignment Adapter for Radiology Report Generation [[paper](https://aclanthology.org/2025.coling-main.571/)]

**NAACL'25**
- DDGIP: Radiology Report Generation Through Disease Description Graph and Informed Prompting [[paper](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-naacl.215.pdf)]
- VividMed: Vision Language Model with Versatile Visual Grounding for Medicine [[paper](https://arxiv.org/pdf/2410.12694)][[code](https://github.com/function2-llx/MMMM)]
- Fact-aware multimodal retrieval augmentation for accurate medical radiology report generation [[paper](https://arxiv.org/abs/2407.15268)][[code](https://github.com/cxcscmu/FactMM-RAG?tab=readme-ov-file)]
- GPT-4V Cannot Generate Radiology Reports Yet [paper](https://aclanthology.org/2025.findings-naacl.113/)]

**MICCAI'25**
- Phrase-Grounded Fact-Checking for Automatically Generated Chest X-Ray Reports [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04981-0_42)]
- Diff-RRG: Longitudinal Disease-Wise Patch Difference as Guidance for LLM-Based Radiology Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04981-0_15)]
- SPEC-CXR: Advancing Clinical Safety Through Entity-Level Performance Evaluation of Chest X-ray Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04981-0_56)]
- Enhancing Radiology Report Interpretation through Modality-Specific RadGraph Fine-Tuning [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04981-0_21)]
- Medical Contrastive Learning of Positive and Negative Mentions [[paper](https://link.springer.com/chapter/10.1007/978-3-032-05141-7_38)]
- TRRG: Towards Truthful Radiology Report Generation With Cross-Modal Disease Clue Enhanced Large Language Models [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04981-0_61)]
- Contrastive Knowledge-Guided Large Language Models for Medical Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04978-0_11)]
- MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_36)]
- RRG-DPO: Direct Preference Optimization for Clinically Accurate Radiology Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_52)]
- Semantic-Aware Chest X-ray Report Generation with Domain-Specific Lexicon and Diversity-Controlled Retrieval [[paper](https://link.springer.com/chapter/10.1007/978-3-032-04978-0_58)][[code](https://github.com/BaochangZhang/DrLS)]
- ITAdaptor: Image-Tag Adapter Framework with Knowledge Enhancement for Radiology Report Generation [[paper](https://papers.miccai.org/miccai-2025/paper/1619_paper.pdf)]

**ICASSP'25**
- A Novel Single Continuous Shot Multiple Lesions Endoscopy Report Generation [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10888634)]
- CoMT: Chain-of-Medical-Thought Reduces Hallucination in Medical Report Generation [[paper](https://arxiv.org/pdf/2406.11451)][[code](https://github.com/FRENKIE-CHIANG/CoMT)]

**ISBI'25**
- R2Gen-Mamba: A Selective State Space Model for Radiology Report Generation [[paper](https://arxiv.org/abs/2410.18135)][[code](https://github.com/YonghengSun1997/R2Gen-Mamba)]
- Prompt-Guided Radiology Report Generation Utilizing SAM [[paper](https://ieeexplore.ieee.org/abstract/document/10980655)]

**MIDL'25**
- Radialog: A large vision-language model for radiology report generation and conversational assistance [[paper](https://github.com/ChantalMP/RaDialog)][[code](https://github.com/ChantalMP/RaDialog)]

**ICMR'25**
- ClearView: A Quality-aware Cross-modal Alignment Framework for CT Report Generation [[paper](https://dl.acm.org/doi/abs/10.1145/3731715.3733287)]

**WACV'25**
- ORID: Organ-Regional Information Driven Framework for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10943499)]

**Radiology'25**
- Privacy-ensuring Open-weights Large Language Models Are Competitive with Closed-weights GPT-4o in Extracting Chest Radiography Findings from Free-Text Reports [[paper](https://pubs.rsna.org/doi/pdf/10.1148/radiol.240895)]
- Diagnostic accuracy and clinical value of a domain-specific multimodal generative AI model for chest radiograph report generation [[paper](https://pubs.rsna.org/doi/full/10.1148/radiol.241476)]
- RadSearch, a Semantic Search Model for Accurate Radiology Report Retrieval with Large Language Model Integration [[paper](https://pubs.rsna.org/doi/epdf/10.1148/radiol.240686)]
- LLMs for Radiology Reports: From General Purpose to Light-Weight Domain Adaptation [[paper](https://pubs.rsna.org/doi/abs/10.1148/radiol.252524)]

**TMM'25**
- Adaptive Medical Topic Learning for Enhanced Fine-grained Cross-modal Alignment in Medical Report Generation[[paper](https://ieeexplore.ieee.org/abstract/document/10891465)]
- Enhancing Radiology Report Generation via Multi-Phased Supervision [[paper](https://ieeexplore.ieee.org/document/11050440)][[code](https://github.com/zailongchen/MultiP-R2Gen)]

**TIP'25**
- Cross-Modal Causal Representation Learning for Radiology Report Generation [[paper](https://pubmed.ncbi.nlm.nih.gov/40378020/)][[code](https://github.com/WissingChen/CMCRL)]
- VTAG: Visual-Textual Association Guided Radiology Reports Generation [[paper](https://ieeexplore.ieee.org/document/11218752)]
- Local Alignment for Medical Vision-Language Pre-training [[paper](https://ieeexplore.ieee.org/document/11237022)]

**TMI'25**
- Spatio-Temporal and Retrieval-Augmented Modeling for Chest X-Ray Report Generation [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10938723)][[code](https://github.com/yangyan22/STREAM)]
- Unlocking the Potential of Weakly Labeled Data: A Co-Evolutionary Learning Framework for Abnormality Detection and Report Generation [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10798513)][[code](https://github.com/jinghanSunn/CoE-DG)]
- Large Language Model with Region-guided Referring and Grounding for CT Report Generation [[[paper](https://arxiv.org/pdf/2411.15539)][[code](https://github.com/zhi-xuan-chen/Reg2RG)]
- A Chain of Diagnosis Framework for Accurate and Explainable Radiology Report Generation [[paper](https://pubmed.ncbi.nlm.nih.gov/40608887/)]
- Enhancing Radiology Report Generation via Multi-Phased Supervision [[paper](https://ieeexplore.ieee.org/document/11050440)][[code](https://github.com/zailongchen/MultiP-R2Gen)]
- Chest X-ray Foundation Model with Global and Local Representations Integration [[paper](https://pubmed.ncbi.nlm.nih.gov/40549526/)][[code](https://github.com/RPIDIAL/CheXFound)]
- Ultrasound Report Generation with Cross-Modality Feature Alignment via Unsupervised Guidance [[paper](https://ieeexplore.ieee.org/document/10599394)][[code](https://github.com/LijunRio/Ultrasound-Report-Generation)]
- Feature Decomposition via Shared Low-rank Matrix Recovery for CT Report Generation [[paper](https://ieeexplore.ieee.org/document/11224350)]

**TNNLS'25**
- OS-RRG: Observation State-Aware Radiology Report Generation With Balanced Diagnosis and Attention Intervention [[paper](https://ieeexplore.ieee.org/document/11095809)]

**MedIA'25**
- Report is a mixture of topics: Topic-guided radiology report generation [[paper](https://www.sciencedirect.com/science/article/pii/S1361841525001331#d1e2923)][[code](https://github.com/chentaohuang/Topic-Guided-Radiology-Report-Generation)]

**Information Fusion'25**
- Enhancing discriminative ability in multimodal LLMs: A contrastive learning approach for CT report generation [[paper](https://www.sciencedirect.com/science/article/pii/S1566253525003136)]

**TBME'25**
- ChatRadio-Valuer: A Chat Large Language Model for Generalizable Radiology Impression Generation on Multi-institution and Multi-system Data [[paper](https://ieeexplore.ieee.org/abstract/document/11122334)]

**ESWA'25**
- Recalibrated cross-modal alignment network for radiology report generation with weakly supervised contrastive learning [[paper](https://www.sciencedirect.com/science/article/pii/S0957417425000168)]
- HKRG: Hierarchical knowledge integration for radiology report generation [[paper](https://www.sciencedirect.com/science/article/pii/S0957417425002441)][[code](https://github.com/LuckyAI-wb/HKRG)]
- RRGMambaFormer: A hybrid Transformer-Mamba architecture for radiology report generation [[paper](https://www.sciencedirect.com/science/article/pii/S0957417425010413#sec4)][[code](https://github.com/lihongzhao99/RRGMambaFormer)]
- Hybrid graph-based radiology report generation [[paper](https://www.sciencedirect.com/science/article/pii/S0957417425019475#absh001)]
- Ultrasound report generation with fuzzy knowledge and multi-modal large language model [[paper](https://www.sciencedirect.com/science/article/pii/S0957417425021748)]
- MedKit: Multi-level Feature Distillation with Knowledge Injection for Radiology Report Generation [[paper](https://www.sciencedirect.com/science/article/pii/S095741742502620X?__cf_chl_tk=oFOtcu7IJJha61Q_hVexYlmUQoA.MdgG5OJHrxCG9FM-1752642219-1.0.1.1-8xBE2YklvaJPQqXhQfFUokGxYK4VXMNdIbxdckSNGao#abs0001)][[code](https://github.com/sujaly/MedKit)]
- DC-RRG: Diagnosis-Centered Cascaded Radiology Report Generation [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425034992)]
- Dynamic feature fusion guiding and multimodal large language model refining for medical image report generation [[paper](https://www.sciencedirect.com/science/article/pii/S095741742503698X)][[code](https://github.com/BearLiX/DFFG-MLLMR)]

**KBS'25**
- Context-enhanced framework for medical image report generation using multimodal contexts [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124015478)][[code](https://github.com/lihongzhao99/Context-Enhanced-Framework)]
- Abnormal-region-aware Multi-modal Feature Fusion for medical report generation [[paper](https://www.sciencedirect.com/science/article/pii/S0950705125005842#sec4)]
- Dual-Level Semantic Collaboration and Inference Network for Medical Image Report Generation [[paper](https://www.sciencedirect.com/science/article/abs/pii/S095070512501319X)]

**JBHI'25**
- Adapter-Enhanced Hierarchical Cross-Modal Pre-training for Lightweight Medical Report Generation [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10856362)][[code](https://github.com/OpenMICG/AHP)]
- Benchmarking Radiology Report Generation from Noisy Free-Texts [[paper](https://ieeexplore.ieee.org/document/11002452)]
- Automatic Radiology Report Generation Based on State-Space Model [[paper](https://ieeexplore.ieee.org/abstract/document/11037239)]
- MFDP: Multi-View Feature Integration and Enhanced Disease Prompting for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/11244798)]

**Neural Networks'25**
- Radiology Report Generation via Visual-Semantic Ambivalence-Aware Network and Focal Self-Critical Sequence Training [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608025009827#preview-section-abstract)]

**Pattern Recognition Letters'25**
- Integrating clinical knowledge and imaging for medical report generation [[paper](https://www.sciencedirect.com/science/article/pii/S0167865525001783#sec4)]

**Meta-Radiology'25**
- S-RRG-Bench: Structured Radiology Report Generation with Fine-Grained Evaluation Framework [[paper](https://www.sciencedirect.com/science/article/pii/S2950162825000396)]

**WWW Companion'25**
- Diversity-Augmented Diffusion Network With LLM Assistance For Radiology Report Generation [[paper](https://dl.acm.org/doi/10.1145/3701716.3717555)]

**arXiv'25**
- GIT-CXR: End-to-End Transformer for Chest X-Ray Report Generation [[paper](https://arxiv.org/pdf/2501.02598)]
- Activating Associative Disease-Aware Vision Token Memory for LLM-Based X-ray Report Generation [[paper](https://arxiv.org/pdf/2501.03458)][[code](https://github.com/Event-AHU/Medical_Image_Analysis/tree/main)]
- RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignment [[paper](https://arxiv.org/abs/2501.07525)]
- MedRAX: Medical Reasoning Agent for Chest X-ray [[paper](https://arxiv.org/pdf/2502.02673)][[code](https://github.com/bowang-lab/MedRAX)]
- Libra: Leveraging Temporal Images for Biomedical Radiology Analysis [[paper](https://arxiv.org/abs/2411.19378)][[code](https://github.com/X-iZhang/Libra)]
- On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation [[paper](https://arxiv.org/abs/2502.19285)]
- CoCa-CXR: Contrastive Captioners Learn Strong Temporal Structures for Chest X-Ray Vision-Language Understanding [[paper](https://arxiv.org/pdf/2502.20509)]
- CheXalign: Preference fine-tuning in chest X-ray interpretation models without human feedback [[paper](https://arxiv.org/pdf/2410.07025)]
- GEMA-Score: Granular Explainable Multi-Agent Score for Radiology Report Evaluation [[paper](https://arxiv.org/pdf/2503.05347)]
- DAgent: A Relational Database-Driven Data Analysis Report Generation Agent [[paper](https://arxiv.org/pdf/2503.13269)]
- LVMedR2: Perception and Reflection-driven Complex Reasoning for Medical Report Generation [[paper](https://arxiv.org/pdf/2504.02885)]
- MedM-VL: What Makes a Good Medical LVLM? [[paper](https://arxiv.org/pdf/2504.04323)][[code](https://github.com/MSIIP/MedM-VL)]
- Leveraging LLMs for Multimodal Retrieval-Augmented Radiology Report Generation via Key Phrase Extraction [[paper](https://arxiv.org/pdf/2504.07415)]
- DualPrompt-MedCap: A Dual-Prompt Enhanced Approach for Medical Image Captioning [[paper](https://arxiv.org/pdf/2504.09598)]
- DART: Disease-aware Image-Text Alignment and Self-correcting Re-alignment for Trustworthy Radiology Report Generation [[paper](https://arxiv.org/pdf/2504.11786)]
- CRG Score: A Distribution-Aware Clinical Metric for Radiology Report Generation [[paper](https://openreview.net/forum?id=18nhqwH1Yq)]
- Reason Like a Radiologist: Chain-of-Thought and Reinforcement Learning for Verifiable Report Generation [[paper](https://arxiv.org/pdf/2504.18453v1)]
- MedPlan: A Two-Stage RAG-Based System for Personalized Medical Plan Generation [[paper](https://arxiv.org/pdf/2503.17900)]
- CBM-RAG: Demonstrating Enhanced Interpretability in Radiology Report Generation with Multi-Agent RAG and Concept Bottleneck Models [[paper](https://arxiv.org/pdf/2504.20898v1)]
- ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification [[paper](https://arxiv.org/pdf/2504.20930v1)]
- MicarVLMoE: A Modern Gated Cross-Aligned Vision-Language Mixture of Experts Model for Medical Image Captioning and Report Generation [[paper](https://arxiv.org/pdf/2504.20343v1)]
- Large-scale Chest Disease Diagnosis Enabled by Multimodal Large Language Models with Self-Supervised Fine-Tuning [[paper](https://www.researchsquare.com/article/rs-6455853/v1)]
- RadRevise: A Benchmark Dataset for Instruction-Based Radiology Report Editing [[paper](https://proceedings.mlr.press/v281/huang25a.html)]
- Evaluating Vision Language Model Adaptations for Radiology Report Generation in Low-Resource Languages [[paper](https://arxiv.org/pdf/2505.01096v1)]
- AOR: Anatomical Ontology-Guided Reasoning for Medical Large Multimodal Model in Chest X-Ray Interpretation [[paper](https://arxiv.org/abs/2505.02830)][[code](https://github.com/Liqq1/AOR)]
- DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation [[paper](https://arxiv.org/pdf/2505.03401v1)][[code](https://github.com/xmed-lab/DDaTR)]
- CheXLearner: Text-Guided Fine-Grained Representation Learning for Progression Detection [[paper](https://arxiv.org/pdf/2505.06903)]
- Describe Anything in Medical Images [[paper](https://arxiv.org/pdf/2505.05804)]
- A Multimodal Multi-Agent Framework for Radiology Report Generation [[paper](https://arxiv.org/pdf/2505.09787v1)]
- Ultrasound Report Generation with Multimodal Large Language Models for Standardized Texts [[paper](https://arxiv.org/pdf/2505.08838)]
- CorBenchX: Large-Scale Chest X-Ray Error Dataset and Vision–Language Model Benchmark for Report Error Correction [[paper](https://arxiv.org/pdf/2505.12057v1)][[code](https://github.com/Liqq1/CorBenchX)]
- Online Iterative Self-Alignment for Radiology Report Generation [[paper](https://arxiv.org/pdf/2505.11983v1)]
- CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays [[paper](https://arxiv.org/pdf/2505.18087)][[code](https://github.com/ttumyche/CXReasonBench)]
- CLEAR: A Clinically-Grounded Tabular Framework for Radiology Report Evaluation [[paper](https://arxiv.org/pdf/2505.16325)]
- Grounding Chest X-Ray Visual Question Answering with Generated Radiology Reports [[paper](https://arxiv.org/pdf/2505.16624)]
- MRGAgents: A Multi-Agent Framework for Improved Medical Report Generation with Med-LVLMs [[paper](https://arxiv.org/pdf/2505.18530)]
- LUNGUAGE: A Benchmark for Structured and Sequential Chest X-ray Interpretation [[paper](https://arxiv.org/pdf/2505.21190)][[code](https://github.com/SuperSupermoon/Lunguage)]
- Look & Mark: Leveraging Radiologist Eye Fixations and Bounding boxes in Multimodal Large Language Models for Chest X-ray Report Generation [[paper](https://arxiv.org/pdf/2505.22222v1)]
- Interpreting Chest X-rays Like a Radiologist: A Benchmark with Clinical Reasoning [[paper](https://arxiv.org/pdf/2505.23143v1)][[code](https://github.com/guanjinquan/CXRTrek)]
- Automated Structured Report Generation [[paper](https://arxiv.org/pdf/2505.24223v1)][[project](https://stanford-aimi.github.io/srrg.html)]
- Structuring Radiology Reports: Challenging LLMs with Lightweight Models [[paper](https://arxiv.org/pdf/2506.00200v1)][[project](https://stanford-aimi.github.io/structuring.html)]
- Evaluating Large Language Models for Zero-Shot Disease Labeling in CT Radiology Reports Across Organ Systems [[paper](https://arxiv.org/pdf/2506.03259v1)]
- ReXVQA: A Large-scale Visual Question Answering Benchmark for Generalist Chest X-ray Understanding [[paper](https://arxiv.org/pdf/2506.04353)]
- Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning [[paper](https://arxiv.org/pdf/2506.07044)][[project](https://alibaba-damo-academy.github.io/lingshu/)]
- Multimodal Large Language Models for Medical Report Generation via Customized Prompt Tuning [[paper](https://arxiv.org/pdf/2506.15477v1)].
- Grounding Chest X-Ray Visual Question Answering with Generated Radiology Reports [[paper](https://arxiv.org/pdf/2505.16624)]
- Recurrent Visual Feature Extraction and Stereo Attentions for CT Report Generation [[paper](https://arxiv.org/pdf/2506.19665)]
- Bridging Vision and Language: Optimal Transport-Driven Radiology Report Generation via LLMs [[paper](https://arxiv.org/abs/2507.03908v1)]
- MedGemma Technical Report [[paper](http://arxiv.org/pdf/2507.05201v1)]
- MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation [[paper](https://arxiv.org/pdf/2507.06992v1)]
- Learnable Retrieval Enhanced Visual-Text Alignment and Fusion for Radiology Report Generation [[paper](https://arxiv.org/pdf/2507.07568)][[code](https://github.com/banbooliang/REVTAF-RRG)]
- A Unified Platform for Radiology Report Generation and Clinician-Centered AI Evaluation [[paper](https://www.medrxiv.org/content/10.1101/2025.07.07.25331018v1#:~:text=In%20this%20study%2C%20we%20conducted%20a%20radiology-focused%20Turing,two%20core%20modules%3A%20Report%20Generation%20and%20Report%20Evaluation.)]
- Semantically Informed Salient Regions Guided Radiology Report Generation [[paper](https://arxiv.org/pdf/2507.11015)]
- CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings [[paper](https://arxiv.org/pdf/2507.17234v1)]
- SURE-Med: Systematic Uncertainty Reduction for Enhanced Reliability in Medical Report Generation [[paper](https://arxiv.org/pdf/2508.01693v1)]
- R2GenKG: Hierarchical Multi-modal Knowledge Graph for LLM-based Radiology Report Generation [[paper](https://arxiv.org/pdf/2508.03426v1)][[code](https://github.com/Event-AHU/Medical_Image_Analysis)]
- Clinically Grounded Agent-based Report Evaluation: An Interpretable Metric for Radiology Report Generation [[paper](https://arxiv.org/pdf/2508.02808v1)]
- PET2Rep: Towards Vision-Language Model-Drived Automated Radiology Report Generation for Positron Emission Tomography [[paper](https://arxiv.org/pdf/2508.04062)]
- Med-GLIP: Advancing Medical Language-Image Pre-training with Large-scale Grounded Dataset [[paper](https://arxiv.org/pdf/2508.10528v1)]
- AMRG: Extend Vision Language Models for Automatic Mammography Report Generation [[paper](https://arxiv.org/pdf/2508.09225)]
- HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks [[paper](https://arxiv.org/pdf/2508.12778v1)]
- PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2508.05353)][[code](https://github.com/mk-runner/PriorRG)]
- Eyes on the Image: Gaze Supervised Multimodal Learning for Chest X-ray Diagnosis and Report Generation [[paper](https://arxiv.org/pdf/2508.13068v1)]
- EchoVLM: Dynamic Mixture-of-Experts Vision-Language Model for Universal Ultrasound Intelligence [[paper](https://arxiv.org/pdf/2509.14977v1)]
- OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation [[paper](https://arxiv.org/pdf/2509.18600v1)]
- Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning [[paper](https://arxiv.org/pdf/2509.19090)]
- RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows [[paper](https://arxiv.org/pdf/2509.20490)]
- Random Direct Preference Optimization for Radiography Report Generation [[paper](https://arxiv.org/pdf/2509.21351v1)]
- Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports [[paper](https://arxiv.org/pdf/2509.21356v1)]
- EditGRPO: Reinforcement Learning with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation [[paper](https://arxiv.org/pdf/2509.22812v1)][[code](https://github.com/taokz/EditGRPO)]
- TemMed-Bench: Evaluating Temporal Medical Image Reasoning in Vision-Language Models [[paper](https://arxiv.org/pdf/2509.25143v1)][[code](https://temmedbench.github.io/)]
- CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding [[paper](https://arxiv.org/pdf/2509.23379v1)]
- Self-Supervised Anatomical Consistency Learning for Vision-Grounded Medical Report Generation [[paper](https://arxiv.org/pdf/2509.25963)]
- Automated Structured Radiology Report Generation with Rich Clinical Context [[paper](https://github.com/vuno/contextualized-srrg)][[code](https://github.com/vuno/contextualized-srrg)]
- Discrete Diffusion Models with MLLMs for Unified Medical Multimodal Generation [[paper](https://arxiv.org/pdf/2510.06131v1)]
- A Review of Longitudinal Radiology Report Generation: Dataset Composition, Methods, and Performance Evaluation [[paper](https://arxiv.org/pdf/2510.12444v1)]
- EMRRG: Efficient Fine-Tuning Pre-trained X-ray Mamba Networks for Radiology Report Generation [[paper](https://arxiv.org/pdf/2510.16776v1)][[code](https://github.com/Event-AHU/Medical_Image_Analysis)]
- CXRAgent: Director-Orchestrated Multi-Stage Reasoning for Chest X-Ray Interpretation [[paper](https://arxiv.org/pdf/2510.21324v1)]
- Reasoning Visual Language Model for Chest X-Ray Analysis [[paper](https://arxiv.org/pdf/2510.23968)][[code](https://github.com/NVIDIA-Medtech)]
- Current Landscape of Automatic Radiology Report Generation with Deep Learning: An Exploratory Systematic Review [[paper](https://www.preprints.org/frontend/manuscript/6461e8366312f9d71b9a431ca4a4d937/download_pub)]
- Medical Report Generation: A Hierarchical Task Structure-Based Cross-Modal Causal Intervention Framework [[paper](https://arxiv.org/pdf/2511.02271v1)]
- UniMedVL: Unifying Medical Multimodal Understanding and Generation through Observation-Knowledge-Analysis [[paper](https://arxiv.org/pdf/2510.15710)][[code](https://github.com/uni-medical/UniMedVL)]
- PETAR: Localized Findings Generation with Mask-Aware Vision-Language Modeling for PET Automated Reporting [[paper](https://arxiv.org/pdf/2510.27680)]
- Radiology Workflow-Guided Hierarchical Reinforcement Fine-Tuning for Medical Report Generation [[paper](https://arxiv.org/pdf/2511.10065v1)]
- A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation [[paper](https://arxiv.org/pdf/2511.12259v1)]
- Boosting Medical Visual Understanding From Multi-Granular Language Learning [[paper](https://openreview.net/forum?id=ccjukmExrB)][[code](https://github.com/HUANGLIZI/MGLL)]
- Closing the Performance Gap Between AI and Radiologists in Chest X-Ray Reporting [[paper](https://arxiv.org/pdf/2511.21735v1)]
- Structure is Supervision: Multiview Masked Autoencoders for Radiology [[paper](https://arxiv.org/pdf/2511.22294v1)]
  
---

### 2024
**Nature Medicine'24**
- Collaboration between clinicians and vision–language models in radiology report generation [[paper](https://www.nature.com/articles/s41591-024-03302-1)]
- A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks [[paper](https://arxiv.org/abs/2305.17100)][[code](https://github.com/taokz/BiomedGPT)]

**Nature Communications'24**
- Enhancing representation in radiography-reports foundation model: a granular alignment algorithm using masked contrastive learning [[paper](https://www.nature.com/articles/s41467-024-51749-0)][[code](https://github.com/SZUHvern/MaCo)]

**NEJM AI'24**
- Towards Generalist Biomedical AI [[paper](https://arxiv.org/pdf/2307.14334)][[code](https://github.com/kyegomez/Med-PaLM/tree/main)]

**AAAI'24**
- Automatic Radiology Reports Generation via Memory Alignment Network [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28279)]
- PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28038)][[code](https://github.com/jhb86253817/PromptMRG)]
- Bootstrapping Large Language Models for Radiology Report Generation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29826)] [[code](https://github.com/synlp/R2-LLM)]

**CVPR'24**
- Instance-level Expert Knowledge and Aggregate Discriminative Attention for Radiology Report Generation [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Bu_Instance-level_Expert_Knowledge_and_Aggregate_Discriminative_Attention_for_Radiology_Report_CVPR_2024_paper.html)] [[code](https://github.com/hnjzbss/EKAGen)]
- AHIVE: Anatomy-aware Hierarchical Vision Encoding for Interactive Radiology Report Retrieval [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_AHIVE_Anatomy-aware_Hierarchical_Vision_Encoding_for_Interactive_Radiology_Report_Retrieval_CVPR_2024_paper.html)] [[code]]
- InVERGe: Intelligent Visual Encoder for Bridging Modalities in Report Generation (**Workshop**) [[paper](https://openaccess.thecvf.com/content/CVPR2024W/MULA/papers/Deria_InVERGe_Intelligent_Visual_Encoder_for_Bridging_Modalities_in_Report_Generation_CVPRW_2024_paper.pdf)][[code]( https://github.com/labsroy007/InVERGe)]
- MedM2G: Unifying Medical Multi-Modal Generation via Cross-Guided Diffusion with Visual Invariant [[paper](https://arxiv.org/pdf/2403.04290)]

**ACL'24**
- DocLens: Multi-aspect Fine-grained Evaluation for Medical Text Generation [[paper](https://arxiv.org/abs/2311.09581)][[code](https://github.com/yiqingxyq/DocLens)]
- SICAR at RRG2024: GPU Poor’s Guide to Radiology Report Generation [[paper](https://aclanthology.org/2024.bionlp-1.55.pdf)]
- BiCAL: Bi-directional Contrastive Active Learning for Clinical Report Generation [[paper](https://aclanthology.org/2024.bionlp-1.25.pdf)]
- CID at RRG24: Attempting in a Conditionally Initiated Decoding of Radiology Report Generation with Clinical Entities [[paper](https://aclanthology.org/2024.bionlp-1.49.pdf)]
- RadGraph-XL: A Large-Scale Expert-Annotated Dataset for Entity and Relation Extraction from Radiology Reports [[paper](https://aclanthology.org/2024.findings-acl.765.pdf#page=10&zoom=100,401,596)][[code](https://github.com/Stanford-AIMI/radgraph)]
- MLeVLM: Improve Multi-level Progressive Capabilities based on Multimodal Large Language Model for Medical Visual Question Answering [[paper](https://aclanthology.org/2024.findings-acl.296.pdf)][[code](https://github.com/RyannChenOO/MLeVLM)]
- Fine-Grained Image-Text Alignment in Medical Imaging Enables Explainable Cyclic Image-Report Generation [[paper](https://aclanthology.org/2024.acl-long.514.pdf)]

**ICLR'24**
- LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation [[paper](https://arxiv.org/abs/2305.11490)][[code](https://github.com/hyn2028/llm-cxr)]

**NIPS'24**
- BenchX: A Unified Benchmark Framework for Medical Vision-Language Pretraining on Chest X-Rays [[paper](https://arxiv.org/pdf/2410.21969)][[code](https://github.com/yangzhou12/BenchX)]
- JRadiEvo: A Japanese Radiology Report Generation Model Enhanced by Evolutionary Optimization of Model Merging (**NIPS Workshop**)[[paper](https://arxiv.org/pdf/2411.09933)]
- Electrocardiogram Report Generation and Question Answering via Retrieval-Augmented Self-Supervised Modeling (**NIPS Workshop**)[[paper](https://openreview.net/forum?id=1oDKH0lox4)]
- Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE [[paper](https://arxiv.org/pdf/2409.17508)][[code](https://github.com/MSIIP/Uni-Med)]
- MediQ: Question-Asking LLMs for Adaptive and Reliable Clinical Reasoning [[paper](https://arxiv.org/pdf/2406.00922)][[code](https://github.com/stellalisy/mediQ/tree/main)]

**ACM MM'24**
- Medical Report Generation via Multimodal Spatio-Temporal Fusion [[paper](https://openreview.net/pdf?id=XKs7DR9GAK)]
- Diffusion Networks with Task-Specific Noise Control for Radiology Report Generation [[paper](https://openreview.net/pdf?id=kbdeQmw2ny)]
- Divide and Conquer: Isolating Normal-Abnormal Attributes in Knowledge Graph-Enhanced Radiology Report Generation [[paper](https://openreview.net/forum?id=TuU8TQVOoj)][[code](https://github.com/ecoxial2007/DCG_Enhanced_distilGPT2)]
- In-context Learning for Zero-shot Medical Report Generation [[paper](https://openreview.net/pdf?id=8zyG2eUgVE)]

**ECCV'24**
- HERGen: Elevating Radiology Report Generation with Longitudinal Data [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07209.pdf)] [[code](https://github.com/fuying-wang/HERGen)]
- Contrastive Learning with Counterfactual Explanations for Radiology Report Generation [[paper](https://arxiv.org/abs/2407.14474)]
- ChEX: Interactive Localization and Region Description in Chest X-rays[[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03114.pdf)][[code](https://github.com/philip-mueller/chex)]
- MedRAT: Unpaired Medical Report Generation via Auxiliary Tasks[[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08947.pdf)][[code](https://github.com/eladhi/MedRAT)]

**EMNLP'24**
- ICON: Improving Inter-Report Consistency of Radiology Report Generation via Lesion-aware Mix-up Augmentation [[paper](https://arxiv.org/abs/2402.12844)] [[code](https://github.com/wjhou/ICon)]
- Divide and Conquer Radiology Report Generation via Observation Level Fine-grained Pretraining and Prompt Tuning [[paper](https://aclanthology.org/2024.emnlp-main.433.pdf)]

**MICCAI'24**
- Textual Inversion and Self-supervised Refinement for Radiology Report Generation [[paper](https://arxiv.org/pdf/2405.20607)] [[code]]
- Structural Entities Extraction and Patient Indications Incorporation for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2405.14905)] [[code](https://github.com/mk-runner/SEI)]
- CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging [[paper](https://arxiv.org/abs/2403.06801)] [[code](https://github.com/ibrahimethemhamamci/CT2Rep)]
- WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole Slide Images [[paper](https://arxiv.org/abs/2311.16480)][[code](https://github.com/cpystan/Wsi-Caption)]
- HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction [[paper](https://arxiv.org/abs/2403.05396)][[data](https://github.com/dddavid4real/HistGen)].
- Multivariate Cooperative Game for Image-Report Pairs: Hierarchical Semantic Alignment for Medical Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_29)]
- MRScore: Evaluating Medical Report with LLM-Based Reward System [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_27)]
- Energy-Based Controllable Radiology Report Generation with Medical Knowledge [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72086-4_23)]
- GMoD: Graph-driven Momentum Distillation Framework with Active Perception of Disease Severity for Radiology Report Generation [[paper](https://papers.miccai.org/miccai-2024/paper/1733_paper.pdf)][[code](https://github.com/xzp9999/GMoD-mian)]
- TiBiX: Leveraging Temporal Information for Bidirectional X-ray and Report Generation (**MICCAI Workshop**)[[paper](https://arxiv.org/pdf/2403.13343)][[code](https://github.com/BioMedIA-MBZUAI/TiBiX)]
- Multivariate Cooperative Game for Image-Report Pairs: Hierarchical Semantic Alignment for Medical Report Generation [[paper](https://papers.miccai.org/miccai-2024/paper/1475_paper.pdf)]
- KARGEN: Knowledge-Enhanced Automated Radiology Report Generation Using Large Language Models [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72086-4_36)]
- Continually Tuning a Large Language Model for Multi-domain Radiology Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72086-4_17)]
- Diagnose with Uncertainty Awareness: Diagnostic Uncertainty Encoding Framework for Radiology Report Generation [[paper](https://link.springer.com/chapter/10.1007/978-3-031-73158-7_4)]


**CIKM'24**
- CLR2G: Cross-modal Contrastive Learning on Radiology Report [[paper](https://dl.acm.org/doi/pdf/10.1145/3627673.3679668)]

**ICASSP'24**
- Improving Radiology Report Generation with D2-Net: When Diffusion Meets Discriminator [[paper](https://ieeexplore.ieee.org/document/10448326)]

**WACV'24**
- Complex Organ Mask Guided Radiology Report Generation [[paper](https://arxiv.org/pdf/2311.02329)][[code](https://github.com/GaryGuTC/COMG_model)]
- CXR-IRGen: An Integrated Vision and Language Model for the Generation of Clinically Accurate Chest X-Ray Image-Report Pairs [[paper](https://openaccess.thecvf.com/content/WACV2024/html/Shentu_CXR-IRGen_An_Integrated_Vision_and_Language_Model_for_the_Generation_WACV_2024_paper.html)][[code](https://github.com/junjie-shentu/CXR-IRGen)]

**ACCV'24**
- FG-CXR: A Radiologist-Aligned Gaze Dataset for Enhancing Interpretability in Chest X-Ray Report Generation [[paper](https://vision.csee.wvu.edu/publications/phamHBPPADNWNL24accv.pdf)][[code](https://github.com/UARK-AICV/FG-CXR)]

**ML4H'24**
- MedAutoCorrect: Image-Conditioned Autocorrection in Medical Reporting [[paper](https://arxiv.org/abs/2412.02971)]

**MedIA'24**
- From Vision to Text: A Comprehensive Review of Natural Image Captioning in Medical Diagnosis and Radiology Report Generation [[paper](https://www.sciencedirect.com/science/article/pii/S1361841524001890)]
- Enhancing the vision–language foundation model with key semantic knowledge-emphasized report refinement [[paper](https://www.sciencedirect.com/science/article/pii/S136184152400224X)]
- DACG: Dual Attention and Context Guidance Model for Radiology Report Generation [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841524003025)][[code](https://github.com/LangWY/DACG)]
- Dual-Modality Visual Feature Flow for Medical Report Generation [[paper](https://www.sciencedirect.com/science/article/pii/S1361841524003384)]

**TMI'24**
- Multi-grained Radiology Report Generation with Sentence-level Image-language Contrastive Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10458706)] [[code]]
- SGT++: Improved Scene Graph-Guided Transformer for Surgical Report Generation [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10330637)][[code]]
- PhraseAug: An Augmented Medical Report Generation Model with Phrasebook [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10560051)] [[code]]
- Token-Mixer: Bind Image and Text in One Embedding Space for Medical Image Reporting [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10552817)] [[code](https://github.com/yangyan22/Token-Mixer)]
- An Organ-aware Diagnosis Framework for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10579857)]
- Attribute Prototype-guided Iterative Scene Graph for Explainable Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10587279)]
- A New Benchmark: Clinical Uncertainty and Severity Aware Labeled Chest X-Ray Images with Multi-Relationship Graph Learning [[paper](https://ieeexplore.ieee.org/abstract/document/10632161)]
- LHR-RFL: Linear Hybrid-Reward based Reinforced Focal Learning for Automatic Radiology Report Generation [[paper](https://ieeexplore.ieee.org/abstract/document/10769570)]
- Unlocking the Potential of Weakly Labeled Data: A Co-Evolutionary Learning Framework for Abnormality Detection and Report Generation [[paper](https://ieeexplore.ieee.org/abstract/document/10798513)][[code](https://github.com/jinghanSunn/CoE-DG)]

**TMM'24**
- Semi-Supervised Medical Report Generation via Graph-Guided Hybrid Feature Consistency [[paper](https://ieeexplore.ieee.org/document/10119200)][[code]]
- Multi-Level Objective Alignment Transformer for Fine-Grained Oral Panoramic X-Ray Report Generation [[paper](https://ieeexplore.ieee.org/document/10443573)][[code]]
- - Knowledge-guided Cross-modal Alignment and Progressive Fusion for Chest X-ray Report Generation [[paper](https://ieeexplore.ieee.org/abstract/document/10814666)]

**JBHI'24**
- CAMANet: Class Activation Map Guided Attention Network for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10400776)] [[code](https://github.com/Markin-Wang/CAMANet)]
- TSGET: Two-Stage Global Enhanced Transformer for Automatic Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10381879)] [[code](https://github.com/Markin-Wang/CAMANet)]
- Eye Gaze Guided Cross-Modal Alignment Network for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/abstract/document/10596697)]

**Expert Systems with Applications'24**
- CheXReport: A transformer-based architecture to generate chest X-ray reports suggestions [[paper](https://www.sciencedirect.com/science/article/pii/S0957417424015112)][[code](https://github.com/felipezeiser/CheXReport)]
- ChatGPT based contrastive learning for radiology report summarization [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424026940#fn2)][[code](https://github.com/jzw1234/Chataug-CCL)]

**Knowledge-Based Systems'24**
- Automatic medical report generation combining contrastive learning and feature difference [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124012644)]
- Context-enhanced framework for medical image report generation using multimodal contexts [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124015478?__cf_chl_tk=XNRc.olaVDjnhJnTWepTvkDVdFSF.zB1psud1BE1.fA-1735525303-1.0.1.1-UA6DRcLBfYu72UCvVi5Z4irCy6hDimukzeMEmmNruVI)][[[code](https://github.com/lz19991122/Context-Enhanced-Framework/tree/main)]

**Neurocomputing'24**
- Improving radiology report generation with multi-grained abnormality prediction [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224008932)]
- An open chest X-ray dataset with benchmarks for automatic radiology report generation in French [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012499)][[data](https://www.casia-cxr.net/)]
- Trust it or not: Confidence-guided automatic radiology report generation [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224001450)]
- VG-CALF: A vision-guided cross-attention and late-fusion network for radiology images in medical visual question answering [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224015017)]


**Academic Radiology'24**
- Practical Evaluation of ChatGPT Performance for Radiology Report Generation [[paper](https://www.sciencedirect.com/science/article/pii/S1076633224004549)]

**Radiology'24**
- Constructing a Large Language Model to Generate Impressions from Findings in Radiology Reports [[paper](https://pubs.rsna.org/doi/pdf/10.1148/radiol.240885)]
- Comparing Commercial and Open-Source Large Language Models for Labeling Chest Radiograph Reports [[paper](https://pubs.rsna.org/doi/epdf/10.1148/radiol.241139)]


**IEEE Transactions on Emerging Topics in Computational Intelligence'24**
- End-to-End Clustering Enhanced Contrastive Learning for Radiology Reports Generation [[paper](https://ieeexplore.ieee.org/abstract/document/10663478)]

**arXiv papers'24**
- Factual Serialization Enhancement: A Key Innovation for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2405.09586)] [[code](https://github.com/mk-runner/FSE)]
- FITA: Fine-grained Image-Text Aligner for Radiology Report Generation [[paper](https://arxiv.org/abs/2405.00962)] [[code]]
- GREEN: Generative Radiology Report Evaluation and Error Notation [[paper](https://arxiv.org/abs/2405.03595)] [[code]]
- CheXpert Plus: Hundreds of Thousands of Aligned Radiology Texts, Images and Patients [[paper](https://arxiv.org/abs/2405.19538)] [[code](https://github.com/Stanford-AIMI/chexpert-plus)]
- Topicwise Separable Sentence Retrieval for Medical Report Generation [[paper](https://arxiv.org/abs/2405.04175)] [[code]]
- Dia-LLaMA: Towards Large Language Model-driven CT Report Generation [[paper](https://arxiv.org/abs/2403.16386)] [[code]]
- MAIRA-2: Grounded Radiology Report Generation [[paper](https://arxiv.org/pdf/2406.04449)][[code]]
- Benchmarking and Boosting Radiology Report Generation for 3D High-Resolution Medical Images [[paper](https://arxiv.org/pdf/2406.07146)]
- The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It [[paper](https://arxiv.org/pdf/2406.13181)][[code](https://anonymous.4open.science/r/anon-D83E/README.md)]
- Improving Expert Radiology Report Summarization by Prompting Large Language Models with a Layperson Summary [[paper](https://arxiv.org/pdf/2406.13181)]
- Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation [[paper](https://arxiv.org/pdf/2407.15268)]
- X-ray Made Simple: Radiology Report Generation and Evaluation with Layman's Terms [[paper](https://arxiv.org/abs/2406.17911)]
- Multi-modal vision-language model for generalizable annotation-free pathology localization and clinical diagnosis [[paper](https://arxiv.org/pdf/2401.02044)][[code](https://github.com/YH0517/AFLoc)]
- Direct Preference Optimization for Suppressing Hallucinated Prior Exams in Radiology Report Generation [[paper]](https://arxiv.org/pdf/2406.06496)]
- R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation [[paper](https://arxiv.org/pdf/2408.09743)][[code](https://github.com/Event-AHU/Medical_Image_Analysis/tree/main/R2GenCSR)]
- Direct Preference Optimization for Suppressing Hallucinated Prior Exams in Radiology Report Generation [[paper](https://arxiv.org/pdf/2406.06496)]
- M4CXR: Exploring Multi-task Potentials of Multi-modal Large Language Models for Chest X-ray Interpretation [[paper](https://arxiv.org/pdf/2408.16213)]
- Medical Report Generation Is A Multi-label Classification Problem [[paper](https://arxiv.org/pdf/2409.00250)]
- KARGEN: Knowledge-enhanced Automated Radiology Report Generation Using Large Language Models [[paper](https://arxiv.org/pdf/2409.05370)]
- Democratizing MLLMs in Healthcare: TinyLLaVA-Med for Efficient Healthcare Diagnostics in Resource-Constrained Settings [[paper](https://arxiv.org/pdf/2409.12184)]
- SLaVA-CXR: Small Language and Vision Assistant for Chest X-ray Report Automation [[paper](https://arxiv.org/pdf/2409.13321)]
- Expert-level vision-language foundation model for real-world radiology and comprehensive evaluation [[paper](https://arxiv.org/pdf/2409.16183)]
- CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset [[paper](https://arxiv.org/pdf/2410.00448)][[code](https://github.com/Event-AHU/Medical_Image_Analysis)]
- 3D-CT-GPT: Generating 3D Radiology Reports through Integration of Large Vision-Language Models [[paper](https://arxiv.org/pdf/2409.19330)]
- Image-aware Evaluation of Generated Medical Reports [[paper](https://arxiv.org/pdf/2410.17357)]
- Text-Enhanced Medical Visual Question Answering [[paper](https://cs231n.stanford.edu/2024/papers/text-enhanced-medical-visual-question-answering.pdf)]
- MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models [[paper](arxiv.org/abs/2410.13085)][[code](https://github.com/richard-peng-xia/MMed-RAG)]
- R2GEN-MAMBA:ASELECTIVESTATESPACEMODELFORRADIOLOGYREPORT GENERATION [[paper](https://arxiv.org/pdf/2410.18135)][[code](https://github.com/YonghengSun1997/R2Gen-Mamba)]
- Uncovering Knowledge Gaps in Radiology Report Generation Models through Knowledge Graphs[[paper](https://arxiv.org/abs/2408.14397)][[code](https://github.com/rajpurkarlab/ReXKG)]
- Diff-CXR: Report-to-CXR generation through a disease-knowledge enhanced diffusion model [[paper](https://arxiv.org/pdf/2410.20165)]
- FINE-GRAINED VERIFIERS: PREFERENCE MODELING AS NEXT-TOKEN PREDICTION IN VISION-LANGUAGE ALIGNMENT [[paper](https://arxiv.org/pdf/2410.14148)]
- Decoding Report Generators: A Cyclic Vision-Language Adapter for Counterfactual Explanations [[paper](https://arxiv.org/pdf/2411.05261)]
- MCL: Multi-view Enhanced Contrastive Learning for Chest X-ray Report Generation [[paper](https://arxiv.org/abs/2411.10224)][[code](https://github.com/mk-runner/MCL)]
- Anatomy-Guided Radiology Report Generation with Pathology-Aware Regional Prompts [[paper](https://arxiv.org/pdf/2411.10789)]
- MAIRA-Seg: Enhancing Radiology Report Generation with Segmentation-Aware Multimodal Large Language Models [[paper](https://arxiv.org/pdf/2411.11362)]
- TRRG: Towards Truthful Radiology Report Generation With Cross-modal Disease Clue Enhanced Large Language Model [[paper](https://arxiv.org/abs/2408.12141)]
- ORID: Organ-Regional Information Driven Framework for Radiology Report Generation [[paper](https://arxiv.org/pdf/2411.13025)]
- ReXrank: A Public Leaderboard for AI-Powered Radiology Report Generation [[paper](https://arxiv.org/pdf/2411.15122)][[code](https://rexrank.ai/)]
- Uncovering Knowledge Gaps in Radiology Report Generation Models through Knowledge Graphs [[paper](https://arxiv.org/pdf/2408.14397)][[code](https://github.com/rajpurkarlab/ReXKG)]
- LaB-RAG: Label Boosted Retrieval Augmented Generation for Radiology Report Generation [[paper](https://arxiv.org/pdf/2411.16523)]
- Large Language Model with Region-guided Referring and Grounding for CT Report Generation [[[paper](https://arxiv.org/pdf/2411.15539)]
- Improving Factuality of 3D Brain MRI Report Generation with Paired Image-domain Retrieval and Text-domain Augmentation [[paper](https://arxiv.org/pdf/2411.15490)]
- MvKeTR: Chest CT Report Generation with Multi-View Perception and Knowledge Enhancement [[paper](https://arxiv.org/pdf/2411.18309)]
- MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware  Multimodal Preference Optimization [[paper](https://arxiv.org/pdf/2412.06141)][[code](https://github.com/aiming-lab/MMedPO)]
- Semantic Consistency-Based Uncertainty Quantification for Factuality in Radiology Report Generation [[paper](https://arxiv.org/pdf/2412.04606)]
- Foundation Models in Radiology: What, How, When, Why and Why Not [[paper](https://arxiv.org/pdf/2411.18730)]
- A Generalist Learner for Multifaceted Medical Image Interpretation [[paper](https://arxiv.org/abs/2405.07988)]
- M4CXR: Exploring Multi-task Potentials of Multi-modal Large Language Models for Chest X-ray Interpretation [[paper](https://arxiv.org/abs/2408.16213)]
- The Impact of AI Assistance on Radiology Reporting: A Pilot Study Using Simulated AI Draft Reports [[paper](https://arxiv.org/pdf/2412.12042)]
- DAMPER: A Dual-Stage Medical Report Generation Framework with Coarse-Grained MeSH Alignment and Fine-Grained Hypergraph Matching [[paper](https://arxiv.org/pdf/2412.14535)]
- X-ray Made Simple: Radiology Report Generation and Evaluation with Layman’s Terms [[paper](https://arxiv.org/pdf/2406.17911v3)][[code](https://github.com/hegehongcha/LaymanRRG)]


---

### 2023
**ICLR'23**
- Advancing radiograph representation learning with masked record modeling [[paper](https://openreview.net/forum?id=w-x7U26GM7j)][[code](https://github.com/RL4M)]

**CVPR'23**
- KiUT: Knowledge-injected U-Transformer for Radiology Report Generation [[paper](https://ieeexplore.ieee.org/document/10203622)] [[code]]
- METransformer: Radiology report generation by transformer with multiple learnable expert tokens [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_METransformer_Radiology_Report_Generation_by_Transformer_With_Multiple_Learnable_Expert_CVPR_2023_paper.html)][[code]]
- Dynamic Graph Enhanced Contrastive Learning for Chest X-Ray Report Generation [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Dynamic_Graph_Enhanced_Contrastive_Learning_for_Chest_X-Ray_Report_Generation_CVPR_2023_paper.html)] [[code](https://github.com/mlii0117/DCL)]
- Interactive and Explainable Region-guided Radiology Report Generation [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Tanida_Interactive_and_Explainable_Region-Guided_Radiology_Report_Generation_CVPR_2023_paper.html)][[code](https://github.com/ttanida/rgrg)]

ICCV'23
- Unify, Align and Refine: Multi-Level Semantic Alignment for Radiology Report Generation [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Unify_Align_and_Refine_Multi-Level_Semantic_Alignment_for_Radiology_Report_ICCV_2023_paper.pdf)]

**ACL'23**
- ORGAN: Observation-Guided Radiology Report Generation via Tree Reasoning [[paper](https://arxiv.org/abs/2306.06466)] [[code](https://github.com/wjhou/ORGan)]

**EMNLP'23**
- RECAP: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning [[paper](https://aclanthology.org/2023.findings-emnlp.140/)] [[code](https://github.com/wjhou/Recap)]
- Normal-Abnormal Decoupling Memory for Medical Report Generation [[paper](https://aclanthology.org/2023.findings-emnlp.131/)] [[code](https://github.com/kzzjk/NADM)]
- Style-Aware Radiology Report Generation with RadGraph and Few-Shot Prompting [[paper](https://arxiv.org/abs/2310.17811)] [[code]]
- PhenotypeCLIP: Phenotype-based Contrastive Learning for Medical Imaging Report Generation [[paper](https://aclanthology.org/2023.emnlp-main.989.pdf)]


**MICCAI'23**
- Utilizing Longitudinal Chest X-Rays and Reports to Pre-Fill Radiology Reports [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_19)] [[code](https://github.com/CelestialShine/Longitudinal-Chest-X-Ray)]


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

**Radiology'23**
- Leveraging GPT-4 for Post Hoc Transformation of Free-text Radiology Reports into Structured Reporting: A Multilingual Feasibility Study [[paper](https://pubs.rsna.org/doi/epdf/10.1148/radiol.230725)]

**Meta-Radiology'23**
- R2gengpt: Radiology report generation with frozen llms [[paper](https://www.sciencedirect.com/science/article/pii/S2950162823000334#bib4)][[code](https://github.com/wang-zhanyu/R2GenGPT)]

**arXiv papers'23**
- MAIRA-1: A specialised large multimodal model for radiology report generation [[paper](https://arxiv.org/abs/2311.13668)] [[code]]
- Longitudinal Data and a Semantic Similarity Reward for Chest X-Ray Report Generation [[paper](https://arxiv.org/abs/2307.09758)][[code](https://github.com/aehrc/cxrmate)]

---

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

---

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

## Survey

- A Systematic Review of Deep Learning-based Research on Radiology Report Generation (**arXiv 2311**) [[paper](https://arxiv.org/abs/2311.14199)]
- A Survey of Deep Learning-based Radiology Report Generation Using Multimodal Data (**arXiv 2405**) [[paper](https://arxiv.org/abs/2405.12833)]
- Automated Radiology Report Generation: A Review of Recent Advances (**IEEE Reviews in Biomedical Engineering'24**) [[paper](https://ieeexplore.ieee.org/abstract/document/10545538)]
- From Vision to Text: A Comprehensive Review of Natural Image Captioning in Medical Diagnosis and Radiology Report Generation (**Medical Image Analysis'24**)[[paper](https://www.sciencedirect.com/science/article/pii/S1361841524001890)]
- Automatic Medical Report Generation: Methods and Applications (**arXiv'2408**) [[paper](https://arxiv.org/pdf/2408.13988)]
- Automatic medical report generation based on deep learning: A state of the art survey (**Computerized Medical Imaging and Graphics'25**)[[paper](https://www.sciencedirect.com/science/article/pii/S0895611124001630)]
- A survey of deep-learning-based radiology report generation using multimodal inputs (**Medical Image Analysis'25**)[[paper](https://www.sciencedirect.com/science/article/pii/S1361841525001744#bib1)]
- Medical radiology report generation: A systematic review of current deep learning methods, trends, and future directions (**Artificial Intelligence in Medicine'25**) [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0933365725001551)]
- Automatic radiology report generation with deep learning: a comprehensive review of methods and advances (Artificial Intelligence Review'25) [[paper](https://link.springer.com/article/10.1007/s10462-025-11337-0)]
- A Review of Longitudinal Radiology Report Generation: Dataset Composition, Methods, and Performance Evaluation [[paper](https://arxiv.org/pdf/2510.12444v1)]

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
- CAD-Chest: Comprehensive Annotation of Diseases based on MIMIC-CXR Radiology Report (**CAD-Chest**)[[data](https://physionet.org/content/cad-chest/1.0/)][[paper](https://ieeexplore.ieee.org/abstract/document/10632161)][[code](https://github.com/MengRes/MIMIC-CAD)]
- VinDr-CXR: An open dataset of chest X-rays with radiologist annotations (**VinDr-CXR**)[[data](https://physionet.org/content/vindr-cxr/1.0.0/)]
- Chest ImaGenome Dataset (**ImaGenome**) [[data](https://physionet.org/content/chest-imagenome/1.0.0/)]
- Interpretable medical image Visual Question Answering via multi-modal relationship graph learning (**Medical-CXR-VQA**) [[**MedIA'24**](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002044)][[code](https://github.com/Holipori/Medical-CXR-VQA)]
- Medical-Diff-VQA: A Large-Scale Medical Dataset for Difference Visual Question Answering on Chest X-Ray Images (**Medical-Diff-VQA**) [[data](https://physionet.org/content/medical-diff-vqa/1.0.0/)][[code](https://github.com/Holipori/MIMIC-Diff-VQA)]
- ReXPref-Prior: A MIMIC-CXR Preference Dataset for Reducing Hallucinated Prior Exams in Radiology Report Generation (**ReXPref-Prior**)[[data](https://www.physionet.org/content/rexpref-prior/1.0.0/)]
- An open chest X-ray dataset with benchmarks for automatic radiology report generation in French (**CASIA-CXR**) [**Neurocomputing'24**] [[data](https://www.casia-cxr.net/)][[paper](https://www.sciencedirect.com/science/article/pii/S0925231224012499#aep-article-footnote-id1)]
- PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology (**WSI-VQA**)[**arXiv'2401**][[paper](https://pathmmu-benchmark.github.io/#/)][[data](https://huggingface.co/datasets/jamessyx/PathMMU)]
- MIMIC-Eye: Integrating MIMIC Datasets with REFLACX and Eye Gaze for Multimodal Deep Learning Applications (**MIMIC-Eye**)[[data](https://physionet.org/content/mimic-eye-multimodal-datasets/1.0.0/#files-panel)][[code](https://github.com/ChihchengHsieh/MIMIC-Eye)]
- PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation (**PadChest-GR**)[[data](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)][[paper](https://ai.nejm.org/doi/full/10.1056/AIdbp2401120)]
- GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis (**GEMeX**)[[paper](https://arxiv.org/pdf/2411.16778)][[project](https://huggingface.co/datasets/BoKelvin/GEMeX)]
- Computed-Tomography-Report-Generation-Datasets (CTRG), including CTRG-Brain-263K and CTRG-Chest-548K [[data](https://github.com/tangyuhao2016/CTRG)]
- Multi-view CXR: A large-scale multi-view benchmark for chest X-ray report generation (**Multi-view CXR**) [[data](https://huggingface.co/datasets/MK-runner/Multi-view-CXR)][[paper](https://arxiv.org/abs/2411.10224)]
- CheXmask Database: a large-scale dataset of anatomical segmentation masks for chest x-ray images (**CheXmask**)[[data](https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/)]
- **LLaVA-Rad MIMIC-CXR** [[data](https://physionet.org/content/llava-rad-mimic-cxr-annotation/1.0.0/)]
- RaDialog Instruct Dataset [[data](https://physionet.org/content/radialog-instruct-dataset/1.1.0/)][[paper](https://arxiv.org/pdf/2311.18681)]
- **M3D-Cap** [[data](https://huggingface.co/datasets/GoodBaiBai88/M3D-Cap)]
- **ReXErr-v1**: Clinically Meaningful Chest X-Ray Report Errors Derived from MIMIC-CXR [[data](https://physionet.org/content/rexerr-v1/1.0.0/)]
- **ReXGradient-160K**: A Large-Scale Publicly Available Dataset of Chest Radiographs with Free-text Reports [[paper](https://arxiv.org/pdf/2505.00228)][[data](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K/tree/main)]
- PARROT: An Open Multilingual Radiology Reports Dataset (**PARROT**) [[paper](https://arxiv.org/abs/2507.22939)][[data](https://github.com/PARROT-reports/PARROT_v1.0)]
- CXR-LT: Multi-Label Long-Tailed Classification on Chest X-Rays (**CXR-LT**)[[dataset](https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/)]

## Metrics
- FineRadScore: A Radiology Report Line-by-Line Evaluation Technique Generating Corrections with Severity Scores (**arXiv'2405**) [[paper](https://arxiv.org/pdf/2405.20613)][[code](https://github.com/rajpurkarlab/FineRadScore)]
- FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation (**EMNLP'23**) [[paper](https://arxiv.org/abs/2305.14251)][[code](https://github.com/shmsw25/FActScore?tab=readme-ov-file)]
- DocLens: Multi-aspect Fine-grained Evaluation for Medical Text Generation (**ACL'24**) [[paper](https://arxiv.org/abs/2311.09581)][[code](https://github.com/yiqingxyq/DocLens)]
- RaTEScore: A Metric for Radiology Report Generation (**EMNLP'24**) [[paper](https://www.medrxiv.org/content/10.1101/2024.06.24.24309405v1)][[code](https://github.com/MAGIC-AI4Med/RaTEScore)][[PyPI](https://pypi.org/project/RaTEScore/)]
- GREEN: Generative Radiology Report Evaluation and Error Notation [[paper](https://arxiv.org/pdf/2405.03595)][[code](https://github.com/Stanford-AIMI/GREEN)]
- When Radiology Report Generation Meets Knowledge Graph (**MIRQI**) [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6989)][[code](https://github.com/xiaosongwang/MIRQI)]
- Evaluating progress in automatic chest X-ray radiology report generation (**RadCliQ**)[[paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5)][[code](https://github.com/rajpurkarlab/CXR-Report-Metric)]
- Evaluating GPT-4 on Impressions Generation in Radiology Reports (**Radiology**)[[paper](https://pubs.rsna.org/doi/full/10.1148/radiol.231259)]
- ReXamine-Global: A Framework for Uncovering Inconsistencies in Radiology Report Generation Metrics (**arXiv'2408**)[[paper](https://arxiv.org/pdf/2408.16208)]
- MRScore: Evaluating Medical Report with LLM-Based Reward System (**MICAAI'24**) [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_27)]
- ER2Score: LLM-based Explainable and Customizable Metric for Assessing Radiology Reports with Reward-Control Loss (**arXiv'2411**)[[paper](https://arxiv.org/pdf/2411.17301)]
- FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models (**arXiv'2411**)[[paper](https://arxiv.org/pdf/2411.18672)]
- A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings (**CheXprompt**)[[paper](https://www.nature.com/articles/s41467-025-58344-x)][[code](https://github.com/microsoft/chexprompt)]
- RadReason: Radiology Report Evaluation Metric with Reasons and Sub-Scores (**RadReason'2508**) [[paper](https://arxiv.org/pdf/2508.15464v1)]
- CREPE: Rapid Chest X-ray Report Evaluation by Predicting Multi-category Error Counts (**EMNLP'25**) [[paper](https://openreview.net/forum?id=gjFuz5jbiJ)][[code](https://github.com/gihuncho/crepe/tree/main)]

## Other Resources
- Learning to Exploit Temporal Structure for Biomedical Vision–Language Processing (**CVPR'23**) [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.pdf)[[code](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)]
- Investigating and Mitigating Object Hallucinations in Pretrained Vision-Language (CLIP) Models [[paper](https://arxiv.org/pdf/2410.03176)][[code](https://github.com/Yufang-Liu/clip_hallucination)]


## Tools
- [CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)
- [coco-caption](https://github.com/tylin/coco-caption)
- [f1chexbert](https://pypi.org/project/f1chexbert/)
- [radgraph](https://pypi.org/project/radgraph/)
- [mimic-cxr](https://github.com/MIT-LCP/mimic-cxr)
- [RaTEScore](https://pypi.org/project/RaTEScore/)
- [torchxrayvision](https://github.com/mlmed/torchxrayvision)
  


## Feel free to reach out to me if you find any interesting papers missing.
email: kangliu422@gmail.com
WeChat: kangliu422
