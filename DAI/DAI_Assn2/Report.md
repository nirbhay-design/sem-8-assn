#### **Nirbhay Sharma (B19CSE114)**
#### **PA-2 Dependable AI**

---

### **Que-1**


**Network Architecture (LSTMCNN)**

<div>
<div class='subimg'>
<img src='report_imgs/MMD.svg' />
</div>
<p>(a) Figure representing LSTMCNN architecture for multimodal classification. The text is first passed to the LSTM network. Correspondingly the image is passed to the CNN network. The final embeddings from both the embeddings are concatenated to pass to the final classification layer.</p>
</div>


**Network Architecture (LSTMCNN + MTL)**

<div>
<div class='subimg'>
<img src='report_imgs/MMD_MML.svg' />
</div>
<p>(a) Figure representing LSTMCNN architecture for multimodal classification. The text is first passed to the LSTM network. Correspondingly the image is passed to the CNN network. The final embeddings from both the embeddings are concatenated to pass to the final classification layer. Simlarly LSTM and CNN embeddings are passed on to the classification layer to get their respective logits. The combination of all the three losses is backpropagated through the network.</p>
</div>

We use two different networks having two different types of CNN backbones. One backbone is Resnet and another is Shufflenet. We perform variety of experiments to analyze the bias associated with each of the network. The first network is represented in Fig 1. which has only one head and the features from CNN and LSTM are concatenated and passed on to the classification layer where $L_{Fus}$ loss is there which is cross entropy loss. The second network is a MTL network which also utilizes the individual features from CNN and LSTM apart from the fusion of their features. There are three loss in this network. First is loss from LSTM features i.e. $L_{LSTM}$. Second is loss from CNN features i.e. $L_{CNN}$ and last one is loss for fusion of features $L_{Fus}$. All the loss functions are cross entropy loss function and the combination of all the three losses are backpropagated through the network. The final loss function $L_{MML}$ of the MTL network is represented as follows:

$$
L_{Fus} = CE(Z_{Fus}, y)
$$

$$
L_{LSTM} = CE(Z_{LSTM}, y)
$$

$$
L_{CNN} = CE(Z_{CNN}, y)
$$

$$
L_{MML} = L_{Fus} + \alpha_1 L_{CNN} + \alpha_2 L_{LSTM}
$$

where $Z_{Fus}$, $Z_{LSTM}$, $Z_{CNN}$ represents the final logits from Fusion layer, LSTM layer, and CNN layer. $y$ represents the true labels of the data points. 

**Average metrics**

To evaluate the network we analyze different networks comprising of different CNN backbones. We use two backbones Resnet and Shufflenet. We report the performance of the models in terms of Average accuracy, Precision, Recall, F1-score, AUROC, DOB (Degree of Bias). 

|Network|Acc|Precision|Recall|F1|AUROC|DOB|
|---|---|---|---|---|---|---|
|LSTMCNN + ShuffleNet|0.87|0.79|0.81|0.80|0.97|0.096|
|LSTMCNN + ResNet|0.84|0.80|0.77|0.78|0.96|0.128|
|LSTMCNN + ShflNet + MTL|0.86|0.66|0.72|0.69|0.97|0.364|
|LSTMCNN + ResNet + MTL|0.90|0.82|0.82|0.82|0.98|0.088|

As presented above the models perform well on the dataset. However a comparison in terms of bias can be observed by using DOB bias metrics. The model having Resnet and trained using MTL learning is showing a low bias as compared to the other. On the other hand the Shufflenet + MTL does not seems to work well in terms of bias in the network. The networks without MTL are also showing promising results. Moreover the LSTMCNN + shufflenet is showing good performance in terms of bias. The networks performs well in terms of other metrics as well. The LSTMCNN + Resnet + MTL is showing a better performance in comparison to other methods in terms of Precision, Recall, F1-Score, and AUROC. Therefore, the least biased among them is LSTMCNN + Resnet + MTL in terms of DOB and other meaningful metrics.

**Accuracy of multiple heads in LSTMCNN**

Next, for MTL network we evaluate the individual heads performance. Additionally, we evaluate the performance in terms of ensemble of the heads. We take the ensemble by taking the average of the logits from each head.  

|Network|Fusion|CNN|LSTM|Ensemble|
|---|---|---|---|---|
|LSTMCNN + ShflNet + MTL|0.856|0.566|0.856|0.858|
|LSTMCNN + ResNet + MTL|0.891|0.689|0.902|0.901|

All the heads are performing individually good. The ensemble of them is also showing promising results and the networks learns better features by backpropagating the combination of three loss values which are represented by $L_{MML}$.


**class wise Precision, Recall, Fscore**

To evaluate more upon the bias. We evaluate class wise precision, recall, F1-score for each of the networks. 

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/lc_res_pr.jpg' />
<p>(a) LSTMCNN_Res</p>
</div>
<div class='subimg'>
<img src='report_imgs/lc_shfl_pr.jpg' />
<p>(b) LSTMCNN_Shfl</p>
</div>
<div class='subimg'>
<img src='report_imgs/lc_res_mtl_pr.jpg' />
<p>(c) LSTMCNN_Res_MTL</p>
</div>
<div class='subimg'>
<img src='report_imgs/lc_shfl_mtl_pr.jpg' />
<p>(d) LSTMCNN_Shfl_MTL</p>
</div>
</div>

As shown in the above illustrations, each model is biased towards different classes. For example considering precision, each models shows different values of precision for each class which implies that they have different kind of learned representation and correspondingly they have different kind of biases in their network.


**Class wise AUROC, DI, and confusion matrix**

We also evaluate bias in terms of class wise AUROC, class wise DI (Disparate Impact), and confusion matrix.

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/lc_res_auc.jpg' />
<p>(a) LSTMCNN_Res</p>
</div>
<div class='subimg'>
<img src='report_imgs/lc_shfl_auc.jpg' />
<p>(b) LSTMCNN_Shfl</p>
</div>
<div class='subimg'>
<img src='report_imgs/lc_res_mtl_auc.jpg' />
<p>(c) LSTMCNN_Res_MTL</p>
</div>
<div class='subimg'>
<img src='report_imgs/lc_shfl_mtl_auc.jpg' />
<p>(d) LSTMCNN_Shfl_MTL</p>
</div>
</div>

Here also different models shows different values for each class. For example we can analyze DI metrics for each of the models class wise. 
- For class 0 LSTMCNN_Res_MTL is showing higher value which implies that among all the models it has least bias towards class 0. 
- For class 1 LSTMCNN_Shfl_MTL is showing higher value which implies that among all the models it has least bias towards class 1.
- For class 2 LSTMCNN_Res is showing higher value which implies that among all the models it has least bias towards class 2.
- For class 3 LSTMCNN_Res_MTL is showing higher value which implies that among all the models it has least bias towards class 3.
- For class 4 LSTMCNN_Shfl_MTL is showing higher value which implies that among all the models it has least bias towards class 4.

This shows that different models shows their bias towards different classes.

**Roc_curves**

We also evaluate the performance using ROC curves. ROC curves are the plots between TPR (True Positive Rate) & FPR (Flase Positive Rate) values and are shown below.

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/lstmcnn_resnet.svg' />
<p>(a) LSTMCNN_Res</p>
</div>
<div class='subimg'>
<img src='report_imgs/lstmcnn_shfl.svg' />
<p>(b) LSTMCNN_Shfl</p>
</div>
<div class='subimg'>
<img src='report_imgs/lstmcnn_MTL_resnet.svg' />
<p>(c) LSTMCNN_Res_MTL</p>
</div>
<div class='subimg'>
<img src='report_imgs/lstmcnn_MTL_shfl.svg' />
<p>(d) LSTMCNN_Shfl_MTL</p>
</div>
</div>

**GradCAM & GradCAM++ Analysis**

<div class='cimg'>
<div class='subimg'>
<img src='gradcam_imgs/gradcam_lstmcnn2orig_5.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2orig_16.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2orig_20.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2orig_27.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2orig_60.png' height=256 width=200/>
<p>(a) Original Images</p>
</div>
<div class='subimg'>
<img src='gradcam_imgs/gradcam_lstmcnn2_5.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2_16.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2_20.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2_27.png' height=256 width=200/>
<img src='gradcam_imgs/gradcam_lstmcnn2_60.png' height=256 width=200/>
<p>(b) GradCAM</p>
</div>
<div class='subimg'>
<img src='gradcampp_imgs/gradcampp_lstmcnn2_5.png' height=256 width=200/>
<img src='gradcampp_imgs/gradcampp_lstmcnn2_16.png' height=256 width=200/>
<img src='gradcampp_imgs/gradcampp_lstmcnn2_20.png' height=256 width=200/>
<img src='gradcampp_imgs/gradcampp_lstmcnn2_27.png' height=256 width=200/>
<img src='gradcampp_imgs/gradcampp_lstmcnn2_60.png' height=256 width=200/>
<p>(c) GradCAM++</p>
</div>
</div>


**Dependency of bias on Network**

Yes, We agree that the bias towards a particular group or class is dependent on the network architecture. As the above experiments depicts that there are four different architectures LSTMCNN+Resnet, LSTMCNN+Shufflenet, LSTMCNN+Resnet+MTL, LSTMCNN+Shufflenet+MTL. Each models has their own specific bias towards a particular class. We can observe it from the metrics values reported above in terms of Precision, Recall, F1-score, AUROC, DI etc. Each network learns different features from the dataset and based on that it tries to develop its understanding of the features which in return generates a variety of bias towards a specific group or a specific class.

**Can explainability be measured in a quantitative way ?**

Yes, we can quantify explainability of a neural network to some extend. Some of the metrics are also proposed in the literature for example SHAP value. This assigns a score based on which feature is important for prediction. Therefore, we can quantify importance of the different features that are being used for output. Similar scores we can provide to each layers via Layer-Wise Relevance Propagation score. Therefore, we can also get a good idea to what layers are most important in prediction of the certain output.

**Another metrics for bias measurement**

We propose another metrics that can be used to measure bias in the model is standard deviation of the precision values. This metrics will tell how far it can deviate in terms of precision. The results of each model on this metrics is as follows:

|Network|Precision std|
|---|---|
|LSTMCNN + ShuffleNet|0.127|
|LSTMCNN + ResNet|0.083|
|LSTMCNN + ShflNet + MTL|0.348|
|LSTMCNN + ResNet + MTL|0.110|

As analyzed using the above metrics LSTMCNN + Resnet shows less bias than others but LSTMCNN + Resnet + MTL is not very far away from the LSTMCNN + Resnet model. But as it is also clear that LSTMCNN + Shfl+ MTL shows a very high bias towards a specific group.


### **Que-2**

**Chossing the paper for Image classification**

For this task we choose Pyramidnet paper [1]. This paper talks about pyramidnet architecture and its implementation. We run it for CIFAR10 dataset which is one the datasets used in the paper. 

<div class='cimg'>
<img src='report_imgs/pyramidnet.JPG'/>
</div>

**Chossing paper for Bias Mitigation**

For bias mitigation we use a paper [2] titled as: **"Permuted AdaIN: Reducing the Bias Towards
Global Statistics in Image Classification"**. The paper talks about textural bias and proposed a normalization layer to mitigate the bias in the network. 

The algorithm (PAdaIN) is represented mathematically as follows.

First defining Instance Normalization (IN). Consider a feature map $x \in R^{N \times C \times H \times W}$. The Instance Norm is defined as:

$$
IN(x) = \gamma (\frac{x-\mu(x)}{\sigma(x)}) + \beta
$$

where $\mu(x)$ and $\sigma(x)$ denotes the mean and standard deviation in the space $R^{N \times C}$. Therefore, defined as:

$$
\mu(x) = \frac{1}{HW} \sum_{h \in [H]} \sum_{w \in [W]} x
$$

$$
\sigma(x) = \sqrt{\frac{1}{HW} \sum_{h \in [H]} \sum_{w \in [W]} (x-\mu(x))^2 + \epsilon}
$$

where we define $[W]$ as an interval of integers ranging from one to W, i.e. $[1,W]$

Then AdaIN is defined over two feature maps. It first tries to normalize $a$ then it tries to scale it up by the statistics of $b$. Therewore it is defined as:

$$
AdaIN(a,b) = \sigma(b) (\frac{a-\mu(a)}{\sigma(a)})  + \mu(b)
$$

where $\mu(a)$ and $\sigma(a)$ are mean and standard deviation of $a$ respectively.

Finally we define permuted AdaIN as follows. Let $x \in R^{N \times C \times H \times W}$ be a feature activation map having components as $x = x_1, x_2, ..., x_N$ and let $\pi(x) = [x_{\pi(1)},x_{\pi(2)},...,x_{\pi(N)}]$ be the permuted feature map after applying a permutation $\pi$. On this permutation and the original feature map we apply $PIN^{\pi}(x)$ operation as follows

$$
PIN^{\pi}(x) = AdaIN(x_i, x_{\pi(i)})
$$

Finally the permuted AdaIN operation on a feature map $x$ is as follows

$$
PAdaIN(x) = (PIN^{\pi}(x_1), PIN^{\pi}(x_2), ..., PIN^{\pi}(x_N))  
$$

The authors have already mentioned in the paper that the PAdaIN is applied after the convolution layer and before the batch-norm layer. The batch-norm layers after the application of PAdaIN does not undo the effect of it. Rather it scales up the output using batch-wise statistics.

**Data techniques for bias mitigation**

Here we apply data augmentations for improving the performance of the architecture. We apply RandomHorizontalFlip and RandomRotation to the data to make it more robust and bias free.

**Model and Loss techniques for bias mitigation**

as represented below we apply MTL approach to mitigate the bias. We use three classification heads having three different loss functions at each head. Each head has different number of neurons to incorporate variations and learn different features. The three losses $L_1$, $L_2$, and $L_3$ are CrossEntropyLoss, KLDivergence Loss, L2Loss functions. The entire loss function is the combination of all these individual heads losses i.e. $L_{MTL} = L_1 + L_2 + L_3$. The combined loss $L_{MTL}$ is backpropagated and trains the network. 

**Network Architecture (MTL)**

<div>
<div class='subimg'>
<img src='report_imgs/MTL.svg' />
</div>
<p>(a) Figure representing MTL architecture for image classification. The image is first passed through the backbone network which is commom for all the heads. The output features are then passed on to the different head continaing different number of neurons to get their respective logits. Three different losses are used at each head and the combination of the losses is backpropagated.</p>
</div>

**Average metrics**

For experiments, we analyze different combination of techniques with pyramidnet. We see the performance of pyramidnet alone, pyramidnet combined with augmentation, pyramidnet combined with PAdaIN, pyramidnet MTL network. We report the performance of the models in terms of Average accuracy, Precision, Recall, F1-score, AUROC, DOB (Degree of Bias). 

|Network|Avg Acc|Avg Precision|Avg Recall|Avg F1|Avg AUROC|DOB|
|---|---|---|---|---|---|---|
|Pyr|0.782|0.79|0.78|0.78|0.972|0.115|
|Pyr + Augment|0.864|0.86|0.86|0.86|0.989|0.087|
|Pyr + PAdaIN + Augment|0.868|0.87|0.87|0.87|0.989|0.075|
|Pyr + MTL + Augment|0.867|0.87|0.87|0.87|0.990|0.069|

As presented above the different approachs works differently on the pyramidnet architecture. The accuracy is highest with Pyr+ PAdaIN approach. However, pyr + MTL is also showing comparable performance to PAdaIN technique. The data augmentation technique is also showing good improvement as compared to the one on which the augmentation is not applied (that shows least accuracy). Similar trends are visible with other metrics as well such as precision, recall, F1-score, AUROC etc. For AUROC pyr + MTL works slightly better than pyr + PAdaIN technique.

**comparison of methods in terms of bias**

In terms of bias metrics (DOB) we observe that bias is least for Pyr + MTL network and is comparable to pyr + PAdaIN technique. The bias for isolated pyramidnet is the most since it is not equipped with techniques like augmentation, MTL etc. 

**performance before and after bias mitigation**

The bias mitigation techniques for sure helps in mitigation of bias in the network. We apply three different techniques to mitigate bias. Augmentation, PAdaIN, MTL techniques are used. after application of each technique the bias is reduced in the network as shown in DOB metrics. PAdaIN also works very well in reducing the bias. Apart from that MTL approach is showing highly comparable performance to PAdaIN approach. So the general trend is decrease in bias as per using different bias mitigation techniques.

**Accuracy of multiple heads in Pyr + MTL + Augment**

Next, we evaluate the accuracy of the individual heads and their ensemble by taking the average logits and applying softmax to that. The reports are available below. All the heads performs equally well and their ensemble is also showing a descent performance and even greater performance than one of the heads.

|Network|Head1|Head2|Head3|Ensemble|
|---|---|---|---|---|
|Pyr + MTL + Augment|0.868|0.868|0.862|0.867|

**class wise Precision, Recall, Fscore**

We also compute bias using class wise metrics such as precision, recall, and F1-score. different techniques combined with pyramidnet shows different values for these metrics for each class. Hence we can compare the models relative to each other by measuring their values classwise only.

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/pyr_pr.jpg' />
<p>(a) Pyr</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_aug_pr.jpg' />
<p>(b) Pyr + Aug</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_padain_pr.jpg' />
<p>(c) Pyr + Aug + PAdaIN</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_mtl_pr.jpg' />
<p>(d) Pyr + Aug + MTL</p>
</div>
</div>

**Class wise DI**

We also report DI (Disparate Impace) class wise of each technique. Since it is reported class wise. We can compare the techniques class wise only. So no network here is absolute best. For instance.

- For class 0, Pyr + PAdaIN is showing less bias
- For class 1, pyr + Augment is showing less bias
- For class 2, pyr is showing less bias
- For class 3, pyr + Aug + MTL is showing less bias
- For class 4, pyr + PAdaIN is showing less bias
- for class 5, pyr + PAdaIN is showing less bias
- for class 6, pyr + Aug + MTL is showing less bias
- for class 7, pyr + Aug + MTL is showing less bias
- for class 8, pyr + Augment is showing less bias
- for class 9, pyr + Aug + MTL is showing less bias

If we compare the methods Pyr + MTL and Pyr + PAdaIN shows quite similar and comparable performance.


<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/pyr_di.jpg' />
<p>(a) Pyr</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_aug_di.jpg' />
<p>(b) Pyr + Aug</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_padain_di.jpg' />
<p>(c) Pyr + Aug + PAdaIN</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_mtl_di.jpg' />
<p>(d) Pyr + Aug + MTL</p>
</div>
</div>

**ROC curves**

We also report ROC curves to see the performance of the models visually.

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/pyr.svg'/>
<p>(a) Pyr</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_augment.svg'/>
<p>(b) Pyr + Aug</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_PAdaIN.svg'/>
<p>(c) Pyr + Aug + PAdaIN</p>
</div>
<div class='subimg'>
<img src='report_imgs/pyr_MTL.svg'/>
<p>(d) Pyr + Aug + MTL</p>
</div>
</div>


**References**

[1] Han, Dongyoon, Jiwhan Kim, and Junmo Kim. "Deep pyramidal residual networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[2] Nuriel, Oren, Sagie Benaim, and Lior Wolf. "Permuted adain: Reducing the bias towards global statistics in image classification." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

<style> 

table, th, td {
  border: 0.1px solid black;
  border-collapse: collapse;
}

.cimg{
    display:flex;
    align-items:center;
    justify-content:center;
}

.subimg{
    text-align:center;
    margin:4px;
}

h3 {
    color: #e71989;
}

</style>

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>