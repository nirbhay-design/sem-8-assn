import torch
import torch.nn as nn
import torch.nn.functional as F
from LstmCNN import *
from que1_lstmcnn import Dataset
import cv2
import yaml
from yaml.loader import SafeLoader
import warnings
warnings.filterwarnings("ignore")

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
        
    return config_data

class GradCAM():
    def __init__(self, model, layer):
        self.model = model
        self.model.eval()
        self.layer = layer
        self.activations = {}
        
        act_layer = self.getlayer(self.model, self.layer)
        act_layer.register_forward_hook(self.forward_hook)
        act_layer.register_backward_hook(self.backward_hook)
        
    def gradcam(self, text, image, label, device, save_path, save_path_orig):
        # image shape [1, 3, H, W]
        
        text = text.to(device)
        image = image.to(device)
        self.model = self.model.to(device)
        
        H,W = image.shape[2], image.shape[3]
        
        out = self.model(text, image)
    
        if len(out) > 1:
            # MTL model
            cnn_feat = out[1]
        else:
            cnn_feat = out
            
        cnn_feat = F.softmax(cnn_feat, dim=1)
        
        _, pred_lbl = cnn_feat.max(dim=1)
        pred_lbl = pred_lbl[0].cpu()
        
        print(f"pred_lbl: {pred_lbl.item()} actual_lbl: {label.item()}")
        
        prob_value = cnn_feat[0, label]
        self.model.zero_grad()
        prob_value.backward()
        
        activations = self.activations['forward']
        grad_activations = self.activations['backward']
        
        gap_output = torch.mean(grad_activations,dim=[2,3])
        go1, go2 = gap_output.shape
        gap_output = gap_output.reshape(go1, go2, 1, 1)
        
        maps = F.relu((gap_output * activations).sum(1, keepdim=True))
        upsample_map = F.upsample(maps, size=(H,W), mode='bilinear', align_corners=False)
        
        min_, max_ = upsample_map.min(), upsample_map.max()
        upsample_map = (upsample_map - min_) / (max_ - min_ + 0.001)
        convert_3_channel = torch.cat([upsample_map, upsample_map, upsample_map],dim=1)
        
        convert_3_channel= convert_3_channel.squeeze().permute(1,2,0).detach().cpu().numpy()
        image = image.squeeze().permute(1,2,0).detach().cpu().numpy()
        
        
        np8_c3c = np.uint8(255 * convert_3_channel)
        np8_orig = np.uint8(image*255)
        
        heatmap = cv2.applyColorMap(np8_c3c, cv2.COLORMAP_JET)
        
        final_img = cv2.addWeighted(np8_orig, 1, heatmap, 0.6, 0)
        final_img = Image.fromarray(final_img)
        final_img.save(save_path)
        orig_img = Image.fromarray(np8_orig)
        orig_img.save(save_path_orig)
        
    def forward_hook(self, model, inp, output):
        self.activations['forward'] = output
        return None
    
    def backward_hook(self, model, grad_inp, grad_out):
        self.activations['backward'] = grad_out[0]
        return None
    
    def getlayer(self, model, layer):
        split_lyr = layer.split('_')
        flayer = model._modules[split_lyr[0]]
        for i in split_lyr[1:]:
            flayer = flayer._modules[i]
        return flayer
    
    
class GradCAMPP():
    def __init__(self, model, layer):
        self.model = model
        self.model.eval()
        self.layer = layer
        self.activations = {}
        
        act_layer = self.getlayer(self.model, self.layer)
        act_layer.register_forward_hook(self.forward_hook)
        act_layer.register_backward_hook(self.backward_hook)
        
    def gradcampp(self, text, image, label, device, save_path, save_path_orig):
        # image shape [1, 3, H, W]
        
        text = text.to(device)
        image = image.to(device)
        self.model = self.model.to(device)
        
        H,W = image.shape[2], image.shape[3]
        
        out = self.model(text, image)
    
        if len(out) > 1:
            # MTL model
            cnn_feat = out[1]
        else:
            cnn_feat = out
            
        cnn_feat = F.softmax(cnn_feat, dim=1)
        
        _, pred_lbl = cnn_feat.max(dim=1)
        pred_lbl = pred_lbl[0].cpu()
        
        print(f"pred_lbl: {pred_lbl.item()} actual_lbl: {label.item()}")
        
        prob_value = cnn_feat[0, label]
        self.model.zero_grad()
        prob_value.backward()
        
        activations = self.activations['forward']
        grad_activations = self.activations['backward']
        
        grad_sqr = grad_activations ** 2
        grad_cub = grad_sqr * grad_activations
        terms = torch.sum(grad_cub * activations, dim=[2,3], keepdim=True)
        alpha = grad_sqr / (2*grad_sqr + terms)
        
        wts = torch.sum(alpha * F.relu(grad_activations), dim=[2,3], keepdim=True)
        
        maps = F.relu((wts * activations).sum(1, keepdim=True))
        upsample_map = F.upsample(maps, size=(H,W), mode='bilinear', align_corners=False)
        
        min_, max_ = upsample_map.min(), upsample_map.max()
        upsample_map = (upsample_map - min_) / (max_ - min_ + 0.001)
        convert_3_channel = torch.cat([upsample_map, upsample_map, upsample_map],dim=1)
        
        convert_3_channel= convert_3_channel.squeeze().permute(1,2,0).detach().cpu().numpy()
        image = image.squeeze().permute(1,2,0).detach().cpu().numpy()
        
        np8_c3c = np.uint8(255 * convert_3_channel)
        np8_orig = np.uint8(image*255)
        
        heatmap = cv2.applyColorMap(np8_c3c, cv2.COLORMAP_JET)
        
        final_img = cv2.addWeighted(np8_orig, 1, heatmap, 0.6, 0)
        final_img = Image.fromarray(final_img)
        final_img.save(save_path)
        orig_img = Image.fromarray(np8_orig)
        orig_img.save(save_path_orig)
        
    def forward_hook(self, model, inp, output):
        self.activations['forward'] = output
        return None
    
    def backward_hook(self, model, grad_inp, grad_out):
        self.activations['backward'] = grad_out[0]
        return None
    
    def getlayer(self, model, layer):
        split_lyr = layer.split('_')
        flayer = model._modules[split_lyr[0]]
        for i in split_lyr[1:]:
            flayer = flayer._modules[i]
        return flayer

if __name__ == "__main__":
    config = yaml_loader(sys.argv[1])
    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    
    print("environment: ")
    print(f"YAML: {sys.argv[1]}")
    for key, value in config.items():
        print(f"==> {key}: {value}")
    
    train_data, test_data, vocab = Dataset(config)
    
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    
    print(len(train_data))
    print(len(test_data))
    print(vocab)
    
    model = LSTMCNN2(
        Flstm=128,
        Fcnn=128,
        Embed_voc=vocab,
        nclass=config['nclass'],
        model=config['cnn_type'],
        device = device
    )
    
    print('==> loading pretrained model')
    print(model.load_state_dict(torch.load(config['saved_path'],map_location=device)))
    
    gradcam = GradCAM(model, config['layer'])
    gradcampp = GradCAMPP(model, config['layer'])
    
    
    text, image, label = next(iter(test_data))
    for batch_idx in range(config['batch_size']):
        
        print(f"batch_idx: {batch_idx}", end=" ")
        
        ntext = text[batch_idx].unsqueeze(0)
        nimage = image[batch_idx].unsqueeze(0)
        nlabel = label[batch_idx]
        
        path = config['gradcam_saved_path'] + f"_{batch_idx}.png"
        orig_path = config['gradcam_saved_path'] + f"orig_{batch_idx}.png"
        
        if config['analysis'] == 'gradcam':
            gradcam.gradcam(ntext, nimage, nlabel, device, path, orig_path)
        else:
            gradcampp.gradcampp(ntext, nimage, nlabel, device, path, orig_path)
            
            
    