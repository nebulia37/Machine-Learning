# Summary of Demos 

## [Demo 00a](../Demos/demo00a_intro_vectors.ipynb)

**New Concepts**: 
* [array creation](https://numpy.org/doc/stable/user/basics.creation.html) 
* [numpy indexing/slicing/striding](https://numpy.org/doc/stable/user/basics.indexing.html) 
* [numpy copies vs views](https://numpy.org/doc/stable/user/basics.copies.html)
* [for loops](https://docs.python.org/3/tutorial/controlflow.html) 
* [printing](https://realpython.com/python-print/)
* [plotting](https://matplotlib.org/stable/users/index.html)

**New Packages**: 
* [numpy](https://numpy.org/doc/stable/user/index.html#user)
* [np.random](https://numpy.org/doc/stable/reference/random/index.html)
* [matplotlib](https://matplotlib.org/stable/users/index.html)

**New Commands**: 
* [np.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html)  
* [np.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
* [np.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
* [print](https://realpython.com/python-print/) 
* [plt.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
* [lsmagic](https://towardsdatascience.com/top-8-magic-commands-in-jupyter-notebook-c1582e813560)

**Dataset**: 
None

---

## [Demo 00b](../Demos/demo00b_python_broadcasting.ipynb)

**New Concepts**: 
* [numpy axis](https://numpy.org/doc/stable/user/absolute_beginners.html?highlight=axis)
* [numpy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

**New Packages**: 
* [numpy](https://numpy.org/doc/stable/user/index.html#user)
* [np.random](https://numpy.org/doc/stable/reference/random/index.html)

**New Commands**: 
* [np.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)
* [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
* `None` or [np.newaxis](https://numpy.org/doc/stable/reference/constants.html?highlight=None#numpy.newaxis)

**Dataset**: 
None

---

## [Demo 01](demo01_auto_mpg.ipynb)

**New Concepts**: 
* loading tabular data 
* handling missing values
* simple linear regression
* simple linear regression on nonlinearly transformed data

**New Packages**: 
* [pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)

**New Commands**: 
* [pd.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

**Dataset**:
[auto-mpg](https://archive.ics.uci.edu/ml/datasets/auto+mpg)

---

## [Demo 02](demo02_diabetes.ipynb)

**New Concepts**: 
* [multiple linear regression](https://scikit-learn.org/stable/modules/linear_model.html)
* [cross-validation (CV)](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

**New Packages**: 
* [sklearn](https://scikit-learn.org/stable/user_guide.html)
* [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
* [sklearn.datasets](https://scikit-learn.org/stable/modules/classes.html?highlight=datasets#module-sklearn.datasets)

**New Commands**: 
* [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [np.hstack](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)
* [np.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
* [np.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)

**Dataset**:
[diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

---

## [Demo 03](demo03_polyfit.ipynb)

**New Concepts**: 
* linear regression of polynomial coefficients
* model order selection
* one-standard-error rule
* [K-fold cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

**New Packages**: 
* [np.polynomial.polynomial](https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html#module-numpy.polynomial.polynomial)
* [sklear.model_selection](https://scikit-learn.org/stable/modules/classes.html?highlight=model%20selection#module-sklearn.model_selection)

**New Commands**: 
* [poly.polyval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyval.html)
* [poly.polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html)
* [plt.errorbar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html)
* [sklear.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
* [np.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html)

**Dataset**:
None

---


## [Demo 04](demo04_prostate.ipynb)

**New Concepts**: 
* [data standardization](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)
* [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)
* [parameter tuning via grid-search](https://scikit-learn.org/stable/modules/grid_search.html#grid-search)
* [LASSO](https://scikit-learn.org/stable/modules/linear_model.html#lasso)
* [Ridge regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
* [Recursive feature elimination (RFE)](https://scikit-learn.org/stable/modules/feature_selection.html#rfe)
* [Univariate feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)
* Feature selection by exhaustive search

**New Packages**: 
* [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
* [sklearn.feature_selection](https://scikit-learn.org/stable/modules/classes.html?highlight=feature%20selection#module-sklearn.feature_selection)
* [sklearn.pipeline](https://scikit-learn.org/stable/modules/compose.html#combining-estimators)

**New Commands**: 
* [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
* [sklearn.model_selection.RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)
* [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
* [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
* best_subset_cv 
* [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
* [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
* [sklearn.linear_model.lasso_path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html)
* LASSO_feature_ranking 
* [np.argmin](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html)
* [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
* [sklearn.feature_selection.RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
* [sklearn.feature_selection.RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
* [sklearn.feature_selection.SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)

**Dataset**:
[prostate data from Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)


---

## [Demo 05a](demo05a_logistic_regression.ipynb)

**New Concepts**: 
* logistic model

**New Packages**: 
* [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)

**New Commands**: 
* [ipywidgets.interact](https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html?highlight=interact)
* [linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

**Dataset**:
None

---

## [Demo 05b](demo05b_breast_cancer.ipynb)

**New Concepts**: 
* [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* [confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)
* [ROC curve](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
* [precision, recall, f1 score](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)

**New Packages**: 
* [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics)

**New Commands**: 
* plot_cnt
* [np.meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html)
* [linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [sklearn.model_selection.cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
* [sklearn.model_selection.cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
* [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
* [sklearn.metrics.ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)
* [sklearn.metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
* [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
* [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
* [sklearn.feature_selection.SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)

**Dataset**:
[breast cancer wisconsin](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

---

## [Demo 06a](demo06a_computing_gradients.ipynb)

**New Concepts**: 
* computing gradients

**New Packages**: 
None

**New Commands**: 
None

**Dataset**:
None

---

## [Demo 06b](demo06b_grad_descent.ipynb)

**New Concepts**: 
* gradient descent
* Armijo rule
* [lambda functions](https://realpython.com/python-lambda/)
* [object-oriented Python](https://realpython.com/python3-object-oriented-programming/)

**New Packages**: 
None

**New Commands**: 
* [lambda](https://docs.python.org/3/reference/expressions.html#lambda)
* [class](https://docs.python.org/3/tutorial/classes.html)

**Dataset**:
[breast cancer wisconsin](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

---

## [Demo 07](demo07_mnist_svm.ipynb)

**New Concepts**: 
* [support vector machine](https://scikit-learn.org/stable/modules/svm.html#svm-classification)

**New Packages**: 
* [sklearn.svm](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
* [pickle](https://realpython.com/python-pickle-module/)

**New Commands**: 
* [sklearn.datasets.fetch_openml](https://scikit-learn.org/stable/datasets/loading_other_datasets.html)
* [plt.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
* [plt.figure](https://matplotlib.org/stable/api/figure_api.html)
* [with](https://www.geeksforgeeks.org/with-statement-in-python/)
* [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
* [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

**Dataset**:
[MNIST](https://en.wikipedia.org/wiki/MNIST_database)

---

## [Demo 08a](demo08a_simple_neural.ipynb)

**New Concepts**: 
* multi-layer perceptron / neural network
* [autograd in PyTorch](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

**New Packages**: 
* [torch](https://pytorch.org/tutorials/)
* [torch.utils.data](https://pytorch.org/docs/stable/data.html)
* [torch.nn](https://pytorch.org/docs/stable/nn.html) (see also this [tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html))
* [torch.optim](https://pytorch.org/docs/stable/optim.html)

**New Commands**: 
* [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)
* [torch.utils.data.TensorDataset](https://pytorch.org/cppdocs/api/structtorch_1_1data_1_1datasets_1_1_tensor_dataset.html)
* [torch.utils.data.DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* [nn.Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid)
* [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
* [nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html?highlight=bceloss#torch.nn.BCELoss)
* [optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html?highlight=optim%20adam)

**Dataset**:
None

---

## [Demo 08b](demo08b_mnist_neural.ipynb)

**New Concepts**: 
* multi-layer perceptron / neural network
* [autograd in PyTorch](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

**New Packages**: 
None 

**New Commands**: 
* [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
* [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropy#torch.nn.CrossEntropyLoss)
* [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html?highlight=torch%20save#torch.save)

**Dataset**:
[MNIST](https://en.wikipedia.org/wiki/MNIST_database)

---

## [Demo 09a](demo09a_cnn_convolutions.ipynb)

**New Concepts**: 
* [2D convolution](https://arxiv.org/pdf/1603.07285.pdf)

**New Packages**: 
* [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
* [skimage.data](https://scikit-image.org/docs/stable/api/skimage.data.html)
* [skimage.io](https://scikit-image.org/docs/stable/api/skimage.io.html)

**New Commands**: 
* disp_image
* [skimage.io.imread](https://scikit-image.org/docs/dev/api/skimage.io#skimage.io.imread)
* [scipy.signal.convolve2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
* [scipy.signal.correlate2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html)
* [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)

**Dataset**:
cameraman and still_life images

---

## [Demo 09b](demo09b_cnn_classifier.ipynb)

**New Concepts**: 
* [Convolutional Neural Network (CNN) classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* dropout
* batch-norm
* data augmentation

**New Packages**: 
* [torchvision](https://pytorch.org/vision/stable/index.html)
* [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html)
* [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
* [optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

**New Commands**: 
* [transforms.ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor)
* [transforms.Normalize](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize)
* [transforms.Compose](https://pytorch.org/vision/stable/transforms.html#compositions-of-transforms)
* [transforms.RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine)
* [transforms.RandomHorizontalFlip](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip)
* [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
* [nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
* [nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
* [nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
* [nn.Dropout2d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html)
* [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

**Dataset**:
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## [Demo 09c](demo09c_cnn_flickr.ipynb)

**New Concepts**: 
* downloading images from Flickr

**New Packages**: 
* [Flickapi](https://stuvel.eu/flickrapi-doc/)
* [os](https://www.geeksforgeeks.org/os-module-python-examples/)

**New Commands**: 
* [flickr.walk](https://stuvel.eu/flickrapi-doc/7-util.html#walking-through-all-photos-in-a-set)

**Dataset**:
Flickr images

---

## [Demo 09d](demo09d_cnn_vgg16.ipynb)

**New Concepts**: 
* VGG16
* transfer learning

**New Packages**: 
* [torchvision.models](https://pytorch.org/vision/stable/models.html)

**New Commands**: 
* [nn.AdaptiveAvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html?highlight=adaptiveavgpool2d#torch.nn.AdaptiveAvgPool2d)
* [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder)
* [transforms.CenterCrop](https://pytorch.org/vision/stable/transforms.html?highlight=centercrop#torchvision.transforms.CenterCrop)
* [torchvision.utils.make_grid](https://pytorch.org/vision/stable/utils.html?highlight=make_grid#torchvision.utils.make_grid)
* [LABELS_URL](https://jbencook.s3.amazonaws.com/pytorch-quick-start/labels.json)

**Dataset**:
Flickr images

---

## [Demo 11](demo11_eigenface_SVM.ipynb)

**New Concepts**: 
* [principal components analysis (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca)
* [eigen-faces](https://en.wikipedia.org/wiki/Eigenface)

**New Packages**: 
* [sklearn.decomposition](https://scikit-learn.org/stable/modules/classes.html?highlight=decomposition#module-sklearn.decomposition)

**New Commands**: 
* [np.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
* [np.linalg.matrix_rank](https://numpy.org/doc/stable/reference/generated/np.linalg.matrix_rank.html)
* [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* [sklearn.metrics.classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

**Dataset**:
[labeled faces in the wild](http://vis-www.cs.umass.edu/lfw/)

---

## [Demo 12a](demo12a_cluster_doc.ipynb)

**New Concepts**: 
* [K-means clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
* [latent semantic analysis (LSA)](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
* [TF-IDF features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* [Non-negative matrix factorization (NMF)](https://scikit-learn.org/stable/modules/decomposition.html#nmf)

**New Packages**:
* [sklearn.feature_extraction](https://scikit-learn.org/stable/modules/classes.html?highlight=feature_extraction#module-sklearn.feature_extraction)
* [sklearn.cluster](https://scikit-learn.org/stable/modules/classes.html?highlight=cluster#module-sklearn.cluster)
* [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)

**New Commands**:
* [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* [scipy.sparse.linalg.svds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)
* [sklearn.decomposition.NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

**Dataset**:
[20 newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)

---

## [Demo 12b](demo12b_cluster_color.ipynb)

**New Concepts**: 
* color quantization
* [Gaussian mixture model (GMM)](https://scikit-learn.org/stable/modules/mixture.html#gmm)
* [expectation-maximization (EM)](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm)

**New Packages**: 
* [sklearn.mixture](https://scikit-learn.org/stable/modules/classes.html?highlight=mixture#module-sklearn.mixture)

**New Commands**: 
* [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

**Dataset**:
china image






