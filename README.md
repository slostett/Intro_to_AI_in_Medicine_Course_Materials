# Intro to AI in Medicine Course Materials


This repository contains the open-source course materials for **Introduction to AI in Medicine**, a course designed to introduce medical students with zero prior experience to foundational concepts, tools, and applications of artificial intelligence in clinical and biomedical contexts.

The course emphasizes conceptual understanding over heavy technical prerequisites, focusing on how AI methods are developed, evaluated, and applied in real medical settings. Topics include data fundamentals, machine learning models, evaluation metrics, bias and ethics, real-world clinical use cases, and a simple mathematical basis for understanding the mechanisms of AI. These materials are intended for educational use and may be adapted for similar courses or self-study.

---

## Repository Contents

This GitHub repository includes:

- 📅 **Course schedule and lecture topics**
- 📊 **Conceptual diagrams** (e.g., the *Four Pillars of AI in Medicine*)
- 📄 **Final project description and guidelines**
- 📁 **Selected student final projects**

Additional materials such as slides, figures, and example notebooks may be included within the corresponding folders.

---

## Course Schedule

The table below outlines the course schedule and core topics covered throughout the term.

**How to use:**

Assignments Column: Homework for each lecture. Students were told to review these links and be ready to discuss during lecture.

Clinical Correlate: Papers or other clinical applications of AI discussing more in-depth in class. No need to read these before lecture.

Supplemental Material: Further reading for those interested in learning topics beyond the scope of the course.


| Lecture Number | Topics | Clinical Correlate | Assignments | Supplemental Material |
|---------------:|--------|-------------------|-------------|----------------------|
| 1 | Course intro, course structure, how to approach this class, what is AI, 4 pillars, formulation of AI problems, history of AI until 1990 | Many | [ELIZA – First Chatbot](https://web.njit.edu/~ronkowit/eliza.html) | |
| 2 | Necessary math for AI: vectors, matrices, tensors, matrix multiplication, inversion, transposition, points, surfaces and functions in n-dimensions, gradients, gradient descent; brief history of modern AI | Atrial fibrillation risk stratification (toy example) | [Matrix multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html) [Matrix multiplication worksheet](https://cdn.kutasoftware.com/Worksheets/Alg2/Matrix%20Multiplication.pdf) | [Matrix multiplication demo](http://matrixmultiplication.xyz/) |
| 3 | Loss functions; linear regression; loss surfaces and gradient descent; local and global minima; mean squared error, cross entropy, domain specific loss; classification problems: ROC curves; logistic regression; AUROC, sensitivity and specificity; supervised vs unsupervised learning; Verifier's law | [Predicting survival in COVID-19 Patients](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01316-6) | [Google lec](https://developers.google.com/machine-learning/crash-course/linear-regression/loss#check-your-understanding) [Gradient descent demo](https://uclaacm.github.io/gradient-descent-visualiser/#playground) [Linear regression](https://mlu-explain.github.io/linear-regression/) | MIT Textbook Ch 2 (dense) |
| 4 | Training: overfitting, cross-validation, revisiting gradient descent, vanishing gradient problem, stochastic GD, hyperparameters, tuning, initialization of weights, pretraining | [Afib ablations w/wo AI: an RCT](https://www.nature.com/articles/s41591-025-03517-w#Sec2) | [Overfitting and underfitting](https://datalab.flitto.com/en/company/blog/what-is-underfitting-and-overfitting-easy-example/) [Neural networks](https://mlu-explain.github.io/neural-networks/) | [Classifiers](https://developers.google.com/machine-learning/crash-course/classification/thresholding) [Karpathy's Training Recipe](https://karpathy.github.io/2019/04/25/recipe/) |
| 5 | Data encodings: one hot, numerical, thermometer encodings; sampling bias; labelling (manual vs automatic); data augmentation; examples of medical data available, how to encode it; constraints of medical data | [Predictors of diabetes](https://www.nature.com/articles/s41598-024-52023-5) [Pneumonia prediction w augmentations](https://colab.research.google.com/github/M-Borsuk/CNN-Pneumonia-Classification/blob/main/PneumoniaScansCNN.ipynb#scrollTo=6gnA0iBtPBcY) [Random CT Windowing as Augmentation](https://arxiv.org/pdf/2510.08116) | [Logistic regression video](https://www.youtube.com/watch?v=EKm0spFxFG4) [ROC Curves](https://mlu-explain.github.io/roc-auc/) | [Logistic regression (dense)](https://mlu-explain.github.io/logistic-regression/) [Classification of abd vs chest x-ray code example](https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb) |
| 6 | Neural networks: review of classical neural net structure; recurrent neural nets, autoencoders, generative adversarial nets (GANs) | | [3b1b neural nets article](https://www.3blue1brown.com/lessons/neural-networks) [Tensorflow demo](https://playground.tensorflow.org/#activation=tanh&regularization=L1&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=8,8&seed=0.64756&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) | MIT Lec – forward/back propagation |
| 8 | Case studies | [Arrhythmia classification](https://www.nature.com/articles/s41591-018-0268-3) [Age from X-ray](https://www.nature.com/articles/s43856-022-00220-6) [Stroke Prediction](https://www.nature.com/articles/s41598-024-82931-5) [Synthetic Bone MRI](https://link.springer.com/article/10.1007/s00330-025-11644-8) [PICU Diagnosis Classification](https://arxiv.org/pdf/1511.03677) | | [Stanford Interactive Demo](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/) |
| 9 | Introduction to computer vision; radiology, dermatology; CNNs; transformer architectures | [Derm lesion classification](https://www.nature.com/articles/nature21056) | [MNIST Demo](https://deeplizard.com/resource/pavq7noze2) [CNN big demo](https://poloclub.github.io/cnn-explainer/) | [1D CNNs for wordle best guess](https://mindthegapblog.com/posts/wordle-solver-ai-cnn/) [CNNs for EKGs](https://www.nature.com/articles/s41591-018-0306-1) |
| 10 | Introduction to large language models/ChatGPT; transformers, preprompting, text mining from notes | [H&N cancer prediction](https://arxiv.org/pdf/2407.07296) | [Language embeddings](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html) [Transformer explained](https://poloclub.github.io/transformer-explainer/) | [High dim embedding space demo](https://projector.tensorflow.org/) [Entropy of Natural Language](https://youtu.be/5eqRuVp65eY) |
| 11 | Guest lecture: AI for Cardiology | Guest lecture | None | |
| 12 | Practicalities of ML: transfer learning, federated learning, hardware (GPU/TPU/ASIC), few shot learning, pre-existing infrastructure (Hugging Face, GitHub, arXiv) | [Transfer learning MR protocols](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321322/) | [Hugging Face](https://huggingface.co/) [arXiv](https://arxiv.org/search/?query=medical&searchtype=all&source=header) [HF Papers](https://huggingface.co/papers/trending?q=medical) [GitHub](https://github.com/search?q=medical&type=repositories) | |
| 13 | AI in the clinic: examples and challenges; EHR systems, PACS, surgical robotics integration; 510(k) FDA medical device classes, "real-time AI", case studies; clinical utility vs accuracy | [EPIC Sepsis Model](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307) | [Google AI Retinal Screen](https://www.technologyreview.com/2020/04/27/1000658/google-medical-ai-accurate-lab-real-life-clinic-covid-diabetes-retina-disease/) [EPIC Sepsis Model](https://news.umich.edu/widely-used-ai-tool-for-early-sepsis-detection-may-be-cribbing-doctors-suspicions/) | |
| 14 | Project introduction | None | | |
| 15 | Fairness & Bias in Medical AI: sources, metrics, case studies | [Predicting Race from Chest X-Ray (full paper)](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext) | [Predicting Race from Medical Imaging](https://news.mit.edu/2022/artificial-intelligence-predicts-patients-race-from-medical-images-0520) Fair ML Textbook pg 205: Case study: the gender earnings gap on Uber | [Fair ML Textbook](https://www.fairmlbook.org/pdf/fairmlbook.pdf) [Bias Types](https://www.chapman.edu/ai/bias-in-ai.aspx) |
| 16 | Ethics of AI: data ownership, likeness generation, environmental concerns; lecture and discussion | [Gender classification](https://news.mit.edu/2018/study-finds-gender-skin-type-bias-artificial-intelligence-systems-0212) | [Reward Tampering (Anthropic AI)](https://www.anthropic.com/research/reward-tampering) | [Proposed ethical AI guidelines](https://link.springer.com/article/10.1007/s11023-018-9482-5) |
| 17 | Interpretable AI: SHAP, LIME, feature importance, saliency maps in imaging | [Saliency maps for CXR interpretation](https://www.nature.com/articles/s42256-022-00536-x) | | [LLM Interpretability](https://aclanthology.org/2024.acl-long.470.pdf) (dense) |
| 18 | Guest lecture: Building a healthcare startup | | | |
| 19 | AI Superintelligence; AI Consciousness; Chinese room allegory; lecture and discussion | | [Race for Superintelligence](https://www.youtube.com/watch?v=5KVDDfAkRgc) | [How does AI self improve?](https://www.technologyreview.com/2025/08/06/1121193/five-ways-that-ai-is-learning-to-improve-itself/) |
| 20 | Guest lecture: Use cases of AI | Guest lecture | | |
| 21 | Sample AI rollout interactive case study | | | |
| 22 | AI Limitations and failures: case studies | | | |
| 23 | **Project Due midnight before - project discussion** | | | |

*(Topics and materials may evolve over time as the course is updated.)*

---

## Conceptual Frameworks

- **Four Pillars of AI in Medicine**  
  A high-level conceptual diagram outlining the core components required for building and evaluating AI systems in medical contexts (e.g., data, models, evaluation, and clinical integration).  
  See: `four_pillars_diagram/`

---

## Final Project

- **Final Project Description**  
  Detailed guidelines for the course final project, including objectives, expectations, and deliverables.  
  See: `Final Project Deliverables.docx` and `Lecture Slides/Lec 12 Project Intro.pptx`

Students are encouraged to explore clinically relevant questions using AI methods discussed in the course, with an emphasis on interpretability, evaluation, and real-world relevance.

---

## Student Projects

Selected student final projects may be included in this repository for reference and educational purposes. Each project is contained in its own folder and typically includes a short description, code or analysis, and results.

See: `Student Final Project Examples/`

---

## Usage Notes

These materials are intended for educational use. They may be reused or adapted with appropriate attribution. Any patient data used in examples or projects is de-identified or synthetic unless otherwise noted.

---

## Contact

For questions about the course or materials, please contact stephenlostetter@yahoo.com.

---
## Works Cited

### Academic Papers & Articles

Banerjee, I., et al. "Reading Race: AI Recognises Patient's Racial Identity in Medical Images." *The Lancet Digital Health*, vol. 4, no. 6, 2022, https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext.

Buolamwini, J., and Gebru, T. "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification." *MIT News*, 11 Feb. 2018, https://news.mit.edu/2018/study-finds-gender-skin-type-bias-artificial-intelligence-systems-0212.

Esteva, A., et al. "Dermatologist-Level Classification of Skin Cancer with Deep Neural Networks." *Nature*, vol. 542, 2017, pp. 115-118, https://www.nature.com/articles/nature21056.

Hannun, A. Y., et al. "Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network." *Nature Medicine*, vol. 25, 2019, pp. 65-69, https://www.nature.com/articles/s41591-018-0268-3.

Hannun, A. Y., et al. "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks." *Nature Medicine*, vol. 25, 2019, https://www.nature.com/articles/s41591-018-0306-1.

Heaven, W. D. "Google's Medical AI Was Super Accurate in a Lab. Real Life Was a Different Story." *MIT Technology Review*, 27 Apr. 2020, https://www.technologyreview.com/2020/04/27/1000658/google-medical-ai-accurate-lab-real-life-clinic-covid-diabetes-retina-disease/.

Heaven, W. D. "Five Ways That AI Is Learning to Improve Itself." *MIT Technology Review*, 6 Aug. 2025, https://www.technologyreview.com/2025/08/06/1121193/five-ways-that-ai-is-learning-to-improve-itself/.

Johnson, A. E. W., et al. "A Comparative Analysis of Sepsis Identification Methods in an Electronic Database." *JAMA Internal Medicine*, vol. 178, no. 11, 2018, pp. 1447-1453, https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307.

Kather, J. N., et al. "Head and Neck Cancer Prediction with Deep Learning." arXiv:2407.07296, 2024, https://arxiv.org/pdf/2407.07296.

Lipton, Z. C., et al. "Learning to Diagnose with LSTM Recurrent Neural Networks." arXiv:1511.03677, 2015, https://arxiv.org/pdf/1511.03677.

Lundberg, S. M., et al. "Explainable Machine-Learning Predictions for the Prevention of Hypoxaemia During Surgery." *Nature Biomedical Engineering*, vol. 6, 2022, pp. 749-760, https://www.nature.com/articles/s42256-022-00536-x.

Moor, M., et al. "Foundation Models for Generalist Medical Artificial Intelligence." *Nature*, vol. 616, 2023, pp. 259-265, https://www.nature.com/articles/s41591-025-03517-w.

Raghu, M., et al. "Transfusion: Understanding Transfer Learning for Medical Imaging." *PMC*, 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC8321322/.

Rajpurkar, P., et al. "AI in Health and Medicine." *Nature Communications*, 2022, https://www.nature.com/articles/s43856-022-00220-6.

Schwab, P., et al. "Predicting Race and Ethnicity from Medical Images." *MIT News*, 20 May 2022, https://news.mit.edu/2022/artificial-intelligence-predicts-patients-race-from-medical-images-0520.

Singh, A., et al. "Synthetic Bone MRI Generation Using Deep Learning." *European Radiology*, 2025, https://link.springer.com/article/10.1007/s00330-025-11644-8.

"Study: Widely Used AI Tool for Early Sepsis Detection May Be Cribbing Doctors' Suspicions." *University of Michigan News*, https://news.umich.edu/widely-used-ai-tool-for-early-sepsis-detection-may-be-cribbing-doctors-suspicions/.

Wang, S., et al. "Predicting Stroke Using Machine Learning." *Scientific Reports*, vol. 14, 2024, https://www.nature.com/articles/s41598-024-82931-5.

Yadav, S., et al. "Predictors of Type 2 Diabetes Using Machine Learning." *Scientific Reports*, vol. 14, 2024, https://www.nature.com/articles/s41598-024-52023-5.

Yan, Y., et al. "COVID-19 Survival Prediction Using Machine Learning." *BMC Medical Informatics and Decision Making*, vol. 20, 2020, https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01316-6.

Zhao, J., et al. "LLM Interpretability: A Survey on Explanation Methods." *ACL Anthology*, 2024, https://aclanthology.org/2024.acl-long.470.pdf.

### Books & Book Chapters

Barocas, S., Hardt, M., and Narayanan, A. *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press, 2023, https://www.fairmlbook.org/pdf/fairmlbook.pdf.

Müller, V. C., and Bostrom, N. "Future Progress in Artificial Intelligence: A Survey of Expert Opinion." *Minds and Machines*, vol. 28, 2018, pp. 555-572, https://link.springer.com/article/10.1007/s11023-018-9482-5.

### Interactive Demos & Educational Tools

"3Blue1Brown: Neural Networks." *3Blue1Brown*, https://www.3blue1brown.com/lessons/neural-networks.

"CNN Explainer: Learn Convolutional Neural Networks." *Georgia Tech Polo Club*, https://poloclub.github.io/cnn-explainer/.

"Classification: Thresholding." *Google Machine Learning Crash Course*, https://developers.google.com/machine-learning/crash-course/classification/thresholding.

"Convolutional Neural Networks for Visual Recognition." *Stanford University*, http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/.

"Embedding Projector." *TensorFlow*, https://projector.tensorflow.org/.

"ELIZA - A Computer Program for the Study of Natural Language Communication." *NJIT*, https://web.njit.edu/~ronkowit/eliza.html.

"Gradient Descent Visualizer." *UCLA ACM*, https://uclaacm.github.io/gradient-descent-visualiser/#playground.

"Linear Regression." *MLU-Explain*, https://mlu-explain.github.io/linear-regression/.

"Logistic Regression." *MLU-Explain*, https://mlu-explain.github.io/logistic-regression/.

"Loss Functions and Optimization." *Google Machine Learning Crash Course*, https://developers.google.com/machine-learning/crash-course/linear-regression/loss.

"Matrix Multiplication." *Math Is Fun*, https://www.mathsisfun.com/algebra/matrix-multiplying.html.

"Matrix Multiplication Demo." *Matrix Multiplication Visualizer*, http://matrixmultiplication.xyz/.

"Matrix Multiplication Worksheet." *Kuta Software*, https://cdn.kutasoftware.com/Worksheets/Alg2/Matrix%20Multiplication.pdf.

"MNIST Demo and Tutorial." *DeepLizard*, https://deeplizard.com/resource/pavq7noze2.

"Neural Networks." *MLU-Explain*, https://mlu-explain.github.io/neural-networks/.

"ROC Curves and AUC Explained." *MLU-Explain*, https://mlu-explain.github.io/roc-auc/.

"TensorFlow Playground." *TensorFlow*, https://playground.tensorflow.org/.

"Transformer Explainer." *Georgia Tech Polo Club*, https://poloclub.github.io/transformer-explainer/.

"Word Embedding Demo." *Carnegie Mellon University*, https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html.

### Code Examples & Notebooks

Borsuk, M. "CNN Pneumonia Classification." *Google Colab*, https://colab.research.google.com/github/M-Borsuk/CNN-Pneumonia-Classification/blob/main/PneumoniaScansCNN.ipynb.

"Lesson 1: X-Ray Images Classification." *MD.ai*, https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb.

### Articles & Blog Posts

Karpathy, A. "A Recipe for Training Neural Networks." *Andrej Karpathy Blog*, 25 Apr. 2019, https://karpathy.github.io/2019/04/25/recipe/.

"1D CNNs for Wordle Best Guess." *Mind the Gap Blog*, https://mindthegapblog.com/posts/wordle-solver-ai-cnn/.

"CT Windowing as Data Augmentation." arXiv:2510.08116, 2024, https://arxiv.org/pdf/2510.08116.

"Overfitting and Underfitting in Machine Learning." *Flitto DataLab*, https://datalab.flitto.com/en/company/blog/what-is-underfitting-and-overfitting-easy-example/.

"What Are the Types of Bias in AI?" *Chapman University*, https://www.chapman.edu/ai/bias-in-ai.aspx.

### Video Content

"Entropy and Information Theory." *YouTube*, https://youtu.be/5eqRuVp65eY.

"Logistic Regression and Maximum Likelihood." *YouTube*, https://www.youtube.com/watch?v=EKm0spFxFG4.

"The Race for AI Superintelligence." *YouTube*, https://www.youtube.com/watch?v=5KVDDfAkRgc.

### Research & Development Platforms

"Anthropic Research: Reward Tampering." *Anthropic*, https://www.anthropic.com/research/reward-tampering.

"arXiv.org Medical Research." *arXiv*, https://arxiv.org/search/?query=medical&searchtype=all&source=header.

"GitHub Medical Repositories." *GitHub*, https://github.com/search?q=medical&type=repositories.

"Hugging Face." *Hugging Face*, https://huggingface.co/.

"Hugging Face Papers - Medical." *Hugging Face*, https://huggingface.co/papers/trending?q=medical.
