# ECE 5307 -- Introduction to Machine Learning 

This repository provides instructional material for machine learning in Python.
The material is used for the Ohio State ECE-5307 Sp23 class taught by
[Prof. Phil Schniter](http://www2.ece.ohio-state.edu/~schniter/)
with [this syllabus](./Misc/ece5307_sp23_syllabus.pdf).

## Course Structure

This one-semester course is taught as a sequence of units, each taking about one or two weeks.
Most units have four components:
* **Lecture**:  Lecture slides in pdf form can be found below.  They will be posted in advance of the in-person lectures from 4:10-5:05pm on MWF in Journalism 251.  In rare cases, these lectures will take place virtually over [Zoom](https://osu.zoom.us/j/93008711267?pwd=MjRnTU9DSkVyUUl4ZzBueFNjTUtxZz09).  Videos will be posted to [YouTube](https://www.youtube.com/playlist?list=PLsN6ERo2QGXJwTlOJ8frNtwAWvVp63iys).
<!--live MWF 8-9am meetings at [https://osu.zoom.us/j/96981482662](https://osu.zoom.us/j/96981482662?pwd=WUFiZU1WdThndFNKazRyZ0dGWkQ4QT09). Links to past lecture videos will also be found below.--> 
* **Demo(s)**: These are Python-based [Jupyter notebooks](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html), found below, that demonstrate concepts from the lectures.  The demos may not cover everything; some details are left for the students to figure out for themselves in the labs.  [Click here for a summary of the demo content.](./Misc/demos.md) 
* **Lab**:  These are Python-based exercises that build on lecture topics and demo code examples.  For each lab, each student will pull a personal GitHub repo using a link found on CarmenCanvas.  The repo will contain data files and a Jupyter notebook with `TODO` sections to fill in.  Once completed, the student will push their repo back to GitHub, where a TA can run the code and grade it.
* **Homework problems**:  These are analytical homework problems, found on CarmenCanvas, to be completed individually by each student.  Each student will upload a pdf of their solutions to CarmenCanvas, for a TA to grade.

## Course Outline
The following outline will grow over the semester...

* Unit 0:  Introduction
    * [Lecture:  Course administration](./Lectures/CourseAdmin.pdf) (mainly covering the [syllabus](./Misc/ece5307_sp23_syllabus.pdf))
    * [Lecture:  What is machine learning?](./Lectures/lect00_introML.pdf)
    * [Demo A:  Introduction to NumPy vectors](./Demos/demo00a_intro_vectors.ipynb)
    * [Demo B:  NumPy broadcasting](./Demos/demo00b_python_broadcasting.ipynb)
    * [Handout:  Linear Algebra Review and Reference](./Misc/cs229-linalg.pdf)

* Unit 1:  Simple linear regression
    * [Lecture:  Simple linear regression](./Lectures/lect01_SimpRegression.pdf)
    * [Demo:  Understanding automobile mpg](./Demos/demo01_auto_mpg.ipynb)

* Unit 2:  Multiple linear regression
    * [Lecture:  Multiple linear regression](./Lectures/lect02_MultLinRegression.pdf)
    * [Demo:  Predicting glucose levels](./Demos/demo02_diabetes.ipynb)

* Unit 3:  Model-order selection
    * [Lecture:  Model-order selection](./Lectures/lect03_ModelSelection.pdf)    
    * [Demo:  Polynomial-order selection with cross-validation](./Demos/demo03_polyfit.ipynb)
    * [Review:  Probability and Expectation](./Misc/expectation.pdf)
    * [Derivation 1:  Bias and variance analysis of linear regression](./Misc/bias_variance_for_linear_regression.pdf)
    * [Derivation 2:  On the bias and variance of sample estimators](./Misc/sample_estimators.pdf)

* Unit 4:  Feature selection and LASSO
    * [Lecture:  Feature selection, LASSO, and Maximum Likelihood](./Lectures/lect04_Lasso.pdf)         
    * [Demo:  Using feature selection to find predictors of prostate cancer](./Demos/demo04_prostate.ipynb) 

* Review 
    * [Review of Units 1-4](./Misc/midterm1_review.pdf)

* Unit 5:  Linear classification and logistic regression
    * [Lecture:  Linear classification and logistic regression](./Lectures/lect05_LogisticReg.pdf)
    * [Demo A:  Logistic regression](./Demos/demo05a_logistic_regression.ipynb)
    * [Demo B:  Breast cancer diagnosis](./Demos/demo05b_breast_cancer.ipynb)

* Unit 6:  Optimization and gradient descent
    * [Lecture:  Nonlinear optimization and gradient descent](./Lectures/lect06_Optim.pdf)
    * [Demo A:  Computing gradients](./Demos/demo06a_computing_gradients.ipynb)
    * [Demo B:  Gradient-descent for logistic regression](./Demos/demo06b_grad_descent.ipynb)

* Unit 7:  Support vector machine
    * [Lecture:  Maximum-margin classification and the support vector machine](./Lectures/lect07_svm.pdf)
    * [Demo:  SVM for MNIST digit recognition](./Demos/demo07_mnist_svm.ipynb)

* Unit 8:  Neural networks
    * [Lecture:  Neural networks](./Lectures/lect08_neural.pdf)
    * [Demo A:  A simple neural network](./Demos/demo08a_simple_neural.ipynb)
    * [Demo B:  MNIST digit recognition](./Demos/demo08b_mnist_neural.ipynb)

* Review 
    * [Review of Units 5-8](./Misc/midterm2_review.pdf)

* Unit 9:  Convolutional and deep neural networks
    * [Lecture:  Convolutional & deep neural networks](./Lectures/lect09_ConvNet.pdf)
    * [Demo A:  2D convolutions](./Demos/demo09a_cnn_convolutions.ipynb)
    * [Demo B:  CIFAR10](./Demos/demo09b_cnn_classifier.ipynb)
    * [Demo C:  Flickr](./Demos/demo09c_cnn_flickr.ipynb)
    * [Demo D:  Pretrained VGG16](./Demos/demo09d_cnn_vgg16.ipynb)

* Unit 10:  Ensemble Methods and Decision Trees 
    * [Lecture:  Random Forests, XGBoost, and other Ensemble Methods](./Lectures/lect10_ensemble.pdf)

* Unit 11:  Principal component analysis (PCA)
    * [Lecture:  Principal component analysis](./Lectures/lect11_pca.pdf)
    * [Demo:  Eigenfaces and classification](./Demos/demo11_eigenface_SVM.ipynb)

* Unit 12:  Clustering, NMF, and EM
    * [Lecture:  Clustering, K-Means, NMF, and EM](./Lectures/lect12_clustering.pdf)
    * [DemoA:  Document clustering and latent semantic analysis](./Demos/demo12a_cluster_doc.ipynb)
    * [DemoB:  Color quantization](./Demos/demo12b_cluster_color.ipynb)

<!--
![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) 
-->

## Important: How to work with Github

The course will use Github to distribute and collect lecture and lab materials.  
* [Downloading/updating course materials from GitHub](./Basics/github.md)
* [Downloading/submitting labs to GitHub](./Basics/github_labs.md)

<!--
* [Instructions to Fork the GitRepository and Make your own update and send pull request](https://github.com/ishjain/learnGithub/blob/master/updateMLrepo.md) 
-->

## Important: How to set up Python and Jupyter

The course will use Python as the programming language and Jupyter notebooks as the environment to run Python in.
You can run them either on your local machine or in the cloud.
<!--
[Amazon web services (AWS)](https://aws.amazon.com),
[Google cloud platform (GCP)](http://cloud.google.com), or
[Microsoft azure](https://azure.microsoft.com).
* [Set up a virtual machine in Google Cloud Platform (GCP)](./GCP/getting_started.md)
-->
* [Instructions to set up Python on your a local machine](./Basics/setup.md) (recommended!)
* Instructions to run Jupyter in the cloud can be found
  [here](https://www.dataschool.io/cloud-services-for-jupyter-notebook/) 
  and
  [here](https://towardsdatascience.com/how-to-run-jupyter-notebooks-in-the-cloud-6ba14ca164da) 
  and
  [here](https://www.exxactcorp.com/blog/Deep-Learning/the-4-best-jupyter-notebook-environments-for-deep-learning).
  (And let me know if you find other good instructions!)

Once you have set up Python, you should be able to run Jupyter notebooks (i.e., files ending in `.ipynb`).  Here are some basic instructions:
* [Running a Jupyter notebook](./Basics/jupyter.md) 

Note that Jupyter notebooks can be _viewed_ in html format on GitHub.com (or with the `nbviewer` browser extension on other websites), but _running_ them requires using Python, either on your local machine or in the cloud. 

<!--
* [Set up a virtual machine in Google Cloud Platform with Docker](./GCP/docker.md)
* [Basics of Python and Its Application for Image Processing through OpenCV](./Basics/PythonTutorial_ACK.pdf)
    * [Example codes and images](./Basics/PythonSampleCodes.zip)
-->

## Optional: Python and NumPy Tutorials

If you are new to Python and NumPy, these tutorials may be helpful.

* [NumPy Quickstart Tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)
* [Python NumPy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)
* [Python i3 Tutorial](https://docs.python.org/3/tutorial/)

If you feel comfortable with Matlab but are new to Python, the following resources may be useful.
* [NumPy for Matlab Users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html) (includes a nice translation table!)
* [A Python Primer for Matlab Users](https://bastibe.de/2013-01-20-a-python-primer-for-matlab-users.html) 
* [Stepping from Matlab to Python](https://stsievert.com/blog/2015/09/01/matlab-to-python/)

If you want some practice with Python and NumPy, here are some exercises (and there are many more on the web):

* [NumPy exercises](https://www.practicaldatascience.org/html/exercises/Exercise_numpy.html)
* [101 NumPy exercises](https://www.machinelearningplus.com/python/101-numpy-exercises-python/)
* [NumPy exercises, practice, solution](https://www.w3resource.com/python-exercises/numpy/index.php)
* [Practice with NumPy arrays (interactive, but need to create a free account)](https://campus.datacamp.com/courses/writing-efficient-python-code/foundations-for-efficiencies?ex=10)
