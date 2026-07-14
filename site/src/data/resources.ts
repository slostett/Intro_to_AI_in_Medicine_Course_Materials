export interface Resource {
  title: string;
  authors?: string;
  publication?: string;
  year?: string;
  url: string;
  type: 'paper' | 'demo' | 'notebook' | 'video' | 'book' | 'platform';
}

export const resources: Resource[] = [
  // Papers
  { title: "Reading Race: AI Recognises Patient's Racial Identity in Medical Images", authors: "Banerjee, I., et al.", publication: "The Lancet Digital Health", year: "2022", url: "https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext", type: "paper" },
  { title: "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification", authors: "Buolamwini, J., and Gebru, T.", publication: "MIT News", year: "2018", url: "https://news.mit.edu/2018/study-finds-gender-skin-type-bias-artificial-intelligence-systems-0212", type: "paper" },
  { title: "Dermatologist-Level Classification of Skin Cancer with Deep Neural Networks", authors: "Esteva, A., et al.", publication: "Nature", year: "2017", url: "https://www.nature.com/articles/nature21056", type: "paper" },
  { title: "Cardiologist-Level Arrhythmia Detection Using a Deep Neural Network", authors: "Hannun, A. Y., et al.", publication: "Nature Medicine", year: "2019", url: "https://www.nature.com/articles/s41591-018-0268-3", type: "paper" },
  { title: "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks", authors: "Hannun, A. Y., et al.", publication: "Nature Medicine", year: "2019", url: "https://www.nature.com/articles/s41591-018-0306-1", type: "paper" },
  { title: "Google's Medical AI Was Super Accurate in a Lab. Real Life Was a Different Story.", authors: "Heaven, W. D.", publication: "MIT Technology Review", year: "2020", url: "https://www.technologyreview.com/2020/04/27/1000658/google-medical-ai-accurate-lab-real-life-clinic-covid-diabetes-retina-disease/", type: "paper" },
  { title: "Five Ways That AI Is Learning to Improve Itself", authors: "Heaven, W. D.", publication: "MIT Technology Review", year: "2025", url: "https://www.technologyreview.com/2025/08/06/1121193/five-ways-that-ai-is-learning-to-improve-itself/", type: "paper" },
  { title: "A Comparative Analysis of Sepsis Identification Methods in an Electronic Database", authors: "Johnson, A. E. W., et al.", publication: "JAMA Internal Medicine", year: "2018", url: "https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307", type: "paper" },
  { title: "Head and Neck Cancer Prediction with Deep Learning", authors: "Kather, J. N., et al.", publication: "arXiv", year: "2024", url: "https://arxiv.org/pdf/2407.07296", type: "paper" },
  { title: "Learning to Diagnose with LSTM Recurrent Neural Networks", authors: "Lipton, Z. C., et al.", publication: "arXiv", year: "2015", url: "https://arxiv.org/pdf/1511.03677", type: "paper" },
  { title: "Explainable Machine-Learning Predictions for the Prevention of Hypoxaemia During Surgery", authors: "Lundberg, S. M., et al.", publication: "Nature Biomedical Engineering", year: "2022", url: "https://www.nature.com/articles/s42256-022-00536-x", type: "paper" },
  { title: "Foundation Models for Generalist Medical Artificial Intelligence", authors: "Moor, M., et al.", publication: "Nature Medicine", year: "2025", url: "https://www.nature.com/articles/s41591-025-03517-w", type: "paper" },
  { title: "Transfusion: Understanding Transfer Learning for Medical Imaging", authors: "Raghu, M., et al.", publication: "PMC", year: "2019", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8321322/", type: "paper" },
  { title: "AI in Health and Medicine", authors: "Rajpurkar, P., et al.", publication: "Nature Communications", year: "2022", url: "https://www.nature.com/articles/s43856-022-00220-6", type: "paper" },
  { title: "Predicting Race and Ethnicity from Medical Images", authors: "Schwab, P., et al.", publication: "MIT News", year: "2022", url: "https://news.mit.edu/2022/artificial-intelligence-predicts-patients-race-from-medical-images-0520", type: "paper" },
  { title: "Synthetic Bone MRI Generation Using Deep Learning", authors: "Singh, A., et al.", publication: "European Radiology", year: "2025", url: "https://link.springer.com/article/10.1007/s00330-025-11644-8", type: "paper" },
  { title: "Study: Widely Used AI Tool for Early Sepsis Detection May Be Cribbing Doctors' Suspicions", publication: "University of Michigan News", url: "https://news.umich.edu/widely-used-ai-tool-for-early-sepsis-detection-may-be-cribbing-doctors-suspicions/", type: "paper" },
  { title: "Predicting Stroke Using Machine Learning", authors: "Wang, S., et al.", publication: "Scientific Reports", year: "2024", url: "https://www.nature.com/articles/s41598-024-82931-5", type: "paper" },
  { title: "Predictors of Type 2 Diabetes Using Machine Learning", authors: "Yadav, S., et al.", publication: "Scientific Reports", year: "2024", url: "https://www.nature.com/articles/s41598-024-52023-5", type: "paper" },
  { title: "COVID-19 Survival Prediction Using Machine Learning", authors: "Yan, Y., et al.", publication: "BMC Medical Informatics and Decision Making", year: "2020", url: "https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01316-6", type: "paper" },
  { title: "LLM Interpretability: A Survey on Explanation Methods", authors: "Zhao, J., et al.", publication: "ACL Anthology", year: "2024", url: "https://aclanthology.org/2024.acl-long.470.pdf", type: "paper" },
  // Books
  { title: "Fairness and Machine Learning: Limitations and Opportunities", authors: "Barocas, S., Hardt, M., and Narayanan, A.", publication: "MIT Press", year: "2023", url: "https://www.fairmlbook.org/pdf/fairmlbook.pdf", type: "book" },
  // Demos
  { title: "ELIZA – First Chatbot", url: "https://web.njit.edu/~ronkowit/eliza.html", type: "demo" },
  { title: "Matrix Multiplication Demo", url: "http://matrixmultiplication.xyz/", type: "demo" },
  { title: "Gradient Descent Visualiser", url: "https://uclaacm.github.io/gradient-descent-visualiser/#playground", type: "demo" },
  { title: "TensorFlow Playground", url: "https://playground.tensorflow.org/", type: "demo" },
  { title: "CNN Explainer", url: "https://poloclub.github.io/cnn-explainer/", type: "demo" },
  { title: "Transformer Explainer", url: "https://poloclub.github.io/transformer-explainer/", type: "demo" },
  { title: "Language Embeddings Demo", url: "https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html", type: "demo" },
  { title: "TensorFlow Projector — Embedding Space", url: "https://projector.tensorflow.org/", type: "demo" },
  { title: "Stanford CS231n Linear Classifier Demo", url: "http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/", type: "demo" },
  { title: "MNIST Demo", url: "https://deeplizard.com/resource/pavq7noze2", type: "demo" },
  // Notebooks
  { title: "Pneumonia Classification with CNN (Colab)", url: "https://colab.research.google.com/github/M-Borsuk/CNN-Pneumonia-Classification/blob/main/PneumoniaScansCNN.ipynb", type: "notebook" },
  { title: "X-Ray Images Classification (Colab)", url: "https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb", type: "notebook" },
  // Videos
  { title: "Logistic Regression (3Blue1Brown)", url: "https://www.youtube.com/watch?v=EKm0spFxFG4", type: "video" },
  { title: "Neural Networks (3Blue1Brown)", url: "https://www.3blue1brown.com/lessons/neural-networks", type: "video" },
  { title: "Entropy of Natural Language", url: "https://youtu.be/5eqRuVp65eY", type: "video" },
  { title: "Race for Superintelligence", url: "https://www.youtube.com/watch?v=5KVDDfAkRgc", type: "video" },
  // Platforms
  { title: "Hugging Face", url: "https://huggingface.co/", type: "platform" },
  { title: "Hugging Face Papers", url: "https://huggingface.co/papers", type: "platform" },
  { title: "arXiv", url: "https://arxiv.org/", type: "platform" },
  { title: "GitHub", url: "https://github.com/", type: "platform" },
  { title: "Google Machine Learning Crash Course", url: "https://developers.google.com/machine-learning/crash-course", type: "platform" },
  { title: "MLU Explain", url: "https://mlu-explain.github.io/", type: "platform" },
];
