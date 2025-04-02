import torch
from utils import mask_circuit


class FinalProject():
    classification_model = ''
    segmentation_model = ''
    threshold = 0.5
    
    def __init__(self, img):
        self.img = img

    def analiz(self):
        class_mod = torch.load(self.classification_model)
        seq_model = torch.load(self.segmentation_model)

        class_mod.eval()
        class_out = class_mod(self.img)
        class_pred = class_out.max(1, keepdim=True)[1]
        
        seq_model.eval()
        seq_out = seq_model(self.img)
        seq_pred = (seq_out > self.threshold).float()

        print(f'')
        return mask_circuit(self.img, seq_pred)


    