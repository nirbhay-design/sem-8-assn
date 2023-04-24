#### **Nirbhay Sharma (B19CSE114)**
#### **Assignment -1**

---
### Installing conda env

```
conda env create -f assn_env.yaml
```

### **Que-1**

**part1**

```
python que1.py configs/que1_train.yaml
```

**part2**

```
python que2.py configs/que2_resnet18.yaml
```

```
python que2.py configs/que2_shufflenet.yaml
```

**part3**

```
python que3.py configs/que3_detnet.yaml
```

### **Que-2**

**For simple binary classification**

```
python que4_simple.py configs/que4_deepfake.yaml
```

**For siamese approach**

```
python que4_siamese.py configs/que4_deepfake.yaml
```

### **Que-3**

**For MelSpectrograms + Siamese approach**

```
python que5_audio_df_siamese.py configs/que5_audio_df.yaml
```

**For Assist**

```
CUDA_VISIBLE_DEVICES=<gpu_id> python main.py --config config/AASIST.conf
```

**Note**: For each of the approaches the paths and hyperparameters need to be changed at the corresponding YAML files


<style> 

table, th, td {
  border: 0.1px solid black;
  border-collapse: collapse;
}


h3 {
    color: #e71989;
}

</style>

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>