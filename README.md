# Intro_to_AI_in_Medicine_Course_Materials


This repository contains the open-source course materials for **Introduction to AI in Medicine**, a course designed to introduce medical students to foundational concepts, tools, and applications of artificial intelligence in clinical and biomedical contexts.

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

| Lecture Number | Topics | Clinical Correlate |
|---------------:|--------|-------------------|
| 1 | Course intro, course structure, how to approach this class, what is AI, four pillars, formulation of AI problems, history of AI until 1990 | Many |
| 2 | Necessary math for AI: vectors, matrices, tensors, matrix operations, points and functions in n-dimensions, gradients, gradient descent; brief history of modern AI | Atrial fibrillation risk stratification (toy example) |
| 3 | Loss functions; linear regression; loss surfaces; local vs global minima; MSE, cross-entropy; classification; ROC curves; logistic regression; AUROC; sensitivity & specificity; supervised vs unsupervised learning; Verifier’s Law | Predicting survival in COVID-19 patients |
| 4 | Training: overfitting, cross-validation, gradient descent revisited, vanishing gradients, stochastic GD, hyperparameters, tuning, initialization, pretraining | AFib ablations with vs without AI (RCT) |
| 5 | Data encodings (one-hot, numerical, thermometer); sampling bias; labeling (manual vs automatic); data augmentation; constraints of medical data | Predictors of diabetes; pneumonia prediction with augmentation; random CT windowing |
| 6 | Neural networks: classical architectures, recurrent neural networks, autoencoders, generative adversarial networks (GANs) | — |
| 8 | Case studies | Arrhythmia classification; age from X-ray; stroke prediction; synthetic bone MRI; PICU diagnosis classification |
| 9 | Introduction to computer vision; radiology and dermatology applications; CNNs; transformer architectures | Dermatologic lesion classification |
| 10 | Introduction to large language models and ChatGPT; transformers; prompting; text mining from clinical notes | Head & neck cancer prediction |
| 11 | Guest lecture | Guest lecture |
| 12 | Practicalities of ML: transfer learning, federated learning, hardware (GPU/TPU/ASIC), few-shot learning, infrastructure (Hugging Face, GitHub, arXiv) | Transfer learning for MR protocols |
| 13 | AI in the clinic: EHRs, PACS, surgical robotics, FDA 510(k), real-time AI, clinical utility vs accuracy | EPIC Sepsis Model |
| 14 | Project introduction | None |
| 15 | Fairness and bias in medical AI: sources, metrics, case studies | Predicting race from chest X-ray |
| 16 | Ethics of AI: data ownership, likeness generation, environmental impact; lecture and discussion | Gender classification |
| 17 | Interpretable AI: SHAP, LIME, feature importance, saliency maps | Saliency maps for chest X-ray interpretation |
| 18 | Guest lecture | Guest lecture |
| 19 | AI superintelligence, AI consciousness, Chinese Room allegory; lecture and discussion | — |
| 20 | Guest lecture | Guest lecture |
| 21 | Sample AI rollout interactive case study | — |
| 22 | AI limitations and failures: case studies | — |
| 23 | Final project discussion (project due midnight before) | — |

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
  See: `final_project/`

Students are encouraged to explore clinically relevant questions using AI methods discussed in the course, with an emphasis on interpretability, evaluation, and real-world relevance.

---

## Student Projects

Selected student final projects may be included in this repository for reference and educational purposes. Each project is contained in its own folder and typically includes a short description, code or analysis, and results.

See: `student_projects/`

---

## Usage Notes

These materials are intended for educational use. They may be reused or adapted with appropriate attribution. Any patient data used in examples or projects is de-identified or synthetic unless otherwise noted.

---

## Contact

For questions about the course or materials, please contact the course instructor or repository maintainers.
