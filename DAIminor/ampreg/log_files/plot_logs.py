import os
import matplotlib.pyplot as plt
import sys

targettxt = sys.argv[1]
targettxt1 = sys.argv[2]
resultsvg = sys.argv[3]
dataset = sys.argv[4]

print(targettxt)
print(targettxt1)

acc = []
with open(targettxt, 'r') as f:
    data = f.read().split('\n')
    for line in data:
        try:
            ln = line.split(',')[-2].split(' ')[-1]
            acc.append(float(ln))
        except:
            pass
        
acc1 = []
with open(targettxt1, 'r') as f:
    data = f.read().split('\n')
    for line in data:
        try:
            ln = line.split(',')[-2].split(' ')[-1]
            acc1.append(float(ln))
        except:
            pass
        
plt.figure(figsize=(5,4))
plt.plot([i for i in range(len(acc))], acc, label='PreResnet')
plt.plot([i for i in range(len(acc1))], acc1, label='WideResnet')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title(f'Test Acc vs Epochs {dataset}')
plt.legend()
plt.grid()
plt.xlim(0,len(acc))
plt.savefig(resultsvg, format='svg')