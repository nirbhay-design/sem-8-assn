#### **Nirbhay Sharma (B19CSE114)**
#### **PA-3 Dependable AI**

---

### **FedAvg Algorithm**

<div class='cimg'>
<div class="subimg">
<img src="report_imgs/fedavg.svg">
<p>(a) Workflow denoting FedAvg algorithm</p>
</div>
</div>

The above illustration shows the workflow of FedAvg algorithm. Each client has its own dataset with it. The server first send its model weight $W$ to each clients. The clients replace their models weights with $W$ as $W_i \leftarrow W$. After replacement they train their models on their respective dataset $D_i$ and update the model weights as $W_i = W_i - \eta \triangledown L(X,Y;W_i)$. Finally they send their respective model weights to server. The server on receiving the weights aggregates them as $W = \frac{1}{n} \sum_{i=1}^{n} W_i$. The aggregated weights are then again send to each client for further communication rounds.

### **Mathematical explanation of the FedAvg function**

The function used for aggregation in Fedavg is as follows:

$$
W = \frac{1}{n} \sum_{i=1}^{n} W_i
$$

This function is the aggregation of the updated weights and biases matrix of a neural network. Each neural network can be basically characterized by its weight and bias matrix for each operation such as convolution etc. The above mathematical operations combines the knowledge of each network by averaging their weight and bias matrix element wise.

### **Experiments**

We perform experiments with Resnet18 (abbreviated as R18) model as the global models. We use three different datasets MNIST, SVHN, Coloured-MNIST (abbreviated as CMNIST) at three different clients having 1000 images per class for a 10 class digit classification task. Client1 holds MNIST dataset, client2 holds SVHN dataset, and client3 holds CMNIST dataset. The test data for each client contains 500 images per class and the test data for server model is the concatenation of the individual clients dataset. We also compare the FedAvg method with the baseline method i.e. training with all the three datasets in centralized fashion. 

### **Results**

**We first report the test accuracy curve for the test dataset at each client and server for their respective datasets. We also report the test accuracy of the centralized training.**

<div class='cimg'>
<div class='subimg'>
<img src='DAI_Assn3/acc_curves/fedavg.svg'/>
<p>(a) FedAvg R18</p>
</div>
<div class='subimg'>
<img src='DAI_Assn3/acc_curves/baseline_r18.svg'/>
<p>(b) Non Fed R18</p>
</div>
</div>

If we see the test accuracy improvement we can observe that in FedAvg case the server and clients accuracy are continuously increasing. In baseline also the test accuracy is increasing as the epochs progress

**We report the test accuracies of server and client models for their test dataset. We also report test accuracies of baseline method.**

|Resnet18 Acc|Server|Client1 (MNIST)|Client2 (SVHN)|Client3 (CMNIST)|Baseline|
|---|---|---|---|---|---|
|Class 0|91|100|88|98|90|
|Class 1|91|99|83|98|93|
|Class 2|88|99|83|98|89|
|Class 3|92|99|69|98|89|
|Class 4|91|99|88|95|88|
|Class 5|83|99|89|95|88|
|Class 6|86|98|87|98|90|
|Class 7|90|98|91|95|92|
|Class 8|92|98|76|96|84|
|Class 9|87|97|84|94|95|
|Avg Acc|89.2|98.6|83.7|96.4|89.9|

In this we can infer that for FedAvg case the accuracy average and class wise accuracy of clients and server seems to be considerable. The aggregated models and the local models have learnt the representations better. For baseline as well it learns better representations and performs well.

**We also report the precision, recall, f1score of server, clients and baseline methods** 

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/c1_pr.JPG'/>
<p>(a) R18 C1 (MNIST)</p>
</div>
<div class='subimg'>
<img src='report_imgs/c2_pr.JPG'/>
<p>(b) R18 C2 (SVHN)</p>
</div>
<div class='subimg'>
<img src='report_imgs/c3_pr.JPG'/>
<p>(c) R18 C3 (CMNIST)</p>
</div>
</div>

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/server_pr.JPG'/>
<p>(d) R18 Server</p>
</div>
<div class='subimg'>
<img src='report_imgs/base_pr.JPG'/>
<p>(e) R18 Baseline</p>
</div>
</div>

For client and server the precision recall for each class is also considerably better and each clients and server shows up promising performance. The baseline method also seems to be working well.

**Confusion matrix is also presented below**

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/c1_cm.JPG'/>
<p>(a) R18 C1 (MNIST)</p>
</div>
<div class='subimg'>
<img src='report_imgs/c2_cm.JPG'/>
<p>(b) R18 C2 (SVHN)</p>
</div>
<div class='subimg'>
<img src='report_imgs/c3_cm.JPG'/>
<p>(c) R18 C3 (CMNIST)</p>
</div>
</div>

<div class='cimg'>
<div class='subimg'>
<img src='report_imgs/server_cm.JPG'/>
<p>(d) R18 Server</p>
</div>
<div class='subimg'>
<img src='report_imgs/base_cm.JPG'/>
<p>(e) R18 Baseline</p>
</div>
</div>

For client and server the confusion matrix is also considerably better and each clients and server shows up promising performance. The baseline method also seems to be working well. Some misclassification in each case has happended but that is very less in number as compared to the True positive predictions which leads to the class wise accuracy good enough.

### **Comparison of FedAvg with Baseline**

As represented in the table There is not a very high difference in FedAvg setup and centralized setup. Please note that in FedAvg setup we don't even send the client's data to the server. However, If we compare closely The centralized accuracy is little bit higher than the FedAvg accuracy. This is due to the fact that a good amount of data is availabe at one place. The difference in Baseline and FedAvg setup will start to show up when data is a bit more non-IID distributed. Here all the clients have around 10000 images which is enough to learn a 10 class classification problem. However if the data is more non-IID distributed then the baseline accuracy will still be the same but the FedAvg accuracy is likely to reduce a little bit. However, for this case both methods are working nicely.

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