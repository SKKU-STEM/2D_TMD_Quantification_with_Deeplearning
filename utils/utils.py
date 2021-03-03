import hyperspy.api as hs
import numpy as np
import torch
import atomap.api as am

from models import *

def image_preprocessing(input_img, device = 'cpu'):
    input_img = (input_img - input_img.data.mean()) / input_img.data.std()
    input_arr = input_img.data
    input_tensor = torch.from_numpy(input_arr)
    input_tensor = input_tensor.view(-1, 1, 
                                     input_img.axes_manager[1].size, 
                                     input_img.axes_manager[0].size)
    input_tensor = input_tensor.to(device)
    input_tensor = input_tensor.float()
    
    return input_tensor

def load_trained_model(material = 'V-WSe2', device = 'cpu', pretrained = True):
    denoising_model = DenoisingNet()
    peak_model = PeakNet()
    classification_model = UNet(n_channels = 1, n_classes = 6, bilinear = True)
    
    if pretrained == True:
        denoising_model.load_state_dict(torch.load('model/trained/%s_denoising.pt' %(material)))
        peak_model.load_state_dict(torch.load('model/trained/%s_peak.pt' %(material)))
        classification_model.load_state_dict(torch.load('model/trained/%s_classification.pt' %(material)))
        denoising_model.eval()
        peak_model.eval()
        classification_model.eval()
    else:
        denoising_model.train()
        peak_model.train()
        classification_model.train()
    
    denoising_model = denoising_model.to(device)
    peak_model = peak_model.to(device)
    classification_model = classification_model.to(device)
        
    return denoising_model, peak_model, classification_model

def from_tensor_to_sig(input_tensor):
    input_arr = input_tensor.cpu().detach().numpy()
    input_arr = input_arr.reshape(input_tensor.shape[-2], input_tensor.shape[-1])
    sig = hs.signals.Signal2D(input_arr)
    
    return sig
    
def from_seg_to_map(class_sig, atom_pos, atom_radius = 7):
    radius = atom_radius
    mapping_arr = np.zeros((class_sig.axes_manager[1].size + radius*2,
                            class_sig.axes_manager[0].size + radius*2))
    
    quant = []
    for i in range(1, 6):
        if i != 5:
            atom_list = atom_pos[np.where(class_sig.data[atom_pos[:, 1], atom_pos[:, 0]] == i)[0]]
        elif i == 5:
            atom_list = atom_pos[np.where((class_sig.data[atom_pos[:, 1], atom_pos[:, 0]] == i)|
                                         (class_sig.data[atom_pos[:, 1], atom_pos[:, 0]] == 0))[0]]
        quant.append(len(atom_list))
        for k in range(-radius, radius+1):
            for l in range(-radius, radius+1):
                if k**2 + l**2 < radius**2:
                    mapping_arr[radius + atom_list[:, 1]+l, radius + atom_list[:, 0]+k] = i
                        
    mapping_sig = hs.signals.Signal2D(mapping_arr)
    mapping_sig.change_dtype('uint8')
    mapping_sig = mapping_sig.isig[radius:-radius, radius:-radius]
    
    return mapping_sig, quant
            
    
def run_model(input_tensor, denoising_model, peak_model, classification_model):
    denoised_output = denoising_model(input_tensor)
    denoised_output = denoising_model(denoised_output)
    peak_output = peak_model(denoised_output)
    classification_output = classification_model(denoised_output)
    classification_output = torch.argmax(classification_output, dim = 1)
    
    denoised_sig = from_tensor_to_sig(denoised_output)
    peak_sig = from_tensor_to_sig(peak_output)
    classification_sig = from_tensor_to_sig(classification_output)
     
    atom_pos = am.get_atom_positions(peak_sig, separation = 5, pca = True)
    
    mapping_sig, quantitative_result = from_seg_to_map(class_sig = classification_sig, 
                                                       atom_pos = atom_pos,
                                                       atom_radius = 7)
    mapping_sig = convert_RGB(mapping_sig)
    
    return denoised_sig, mapping_sig, quantitative_result

def convert_RGB(mapping_sig):
    mapping_arr = np.zeros((mapping_sig.axes_manager[1].size, mapping_sig.axes_manager[0].size, 3))
    ind = np.where(mapping_sig.data == 1)
    mapping_arr[ind[0], ind[1], 0] = 255

    ind = np.where(mapping_sig.data == 2)
    mapping_arr[ind[0], ind[1], 0] = 255
    mapping_arr[ind[0], ind[1], 1] = 255

    ind = np.where(mapping_sig.data == 3)
    mapping_arr[ind[0], ind[1], 1] = 128

    ind = np.where(mapping_sig.data == 4)
    mapping_arr[ind[0], ind[1], 1] = 255
    mapping_arr[ind[0], ind[1], 2] = 255

    ind = np.where(mapping_sig.data == 5)
    mapping_arr[ind[0], ind[1], 2] = 255
    
    sig = hs.signals.Signal1D(mapping_arr)
    sig.change_dtype('uint8')
    sig.change_dtype('rgb8')
    
    return sig

def print_quantitative_information(quantitative_result, material):
    print('======Quantitative Information======')
    if material == 'V-WSe2':
        print('W : %d' %(quantitative_result[0]))
        print('V : %d' %(quantitative_result[1]))
        print('Se2 : %d' %(quantitative_result[2]))
        print('Se Single Vacancy : %d' %(quantitative_result[3]))
        print('Se Double Vacancy : %d' %(quantitative_result[4]))
    elif material == 'V-MoS2':
        print('Mo : %d' %(quantitative_result[0]))
        print('V : %d' %(quantitative_result[1]))
        print('S2 : %d' %(quantitative_result[2]))
        print('S Single Vacancy : %d' %(quantitative_result[3]))
        print('S Double Vacancy : %d' %(quantitative_result[4]))
