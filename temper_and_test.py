import models.models as clmgnet
import random
import torch
import copy
import os


def random_select_sample(path, num):
    file_list = []
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if not os.path.isdir(os.path.join(root, d))]
        for file in files:
            file_path = os.path.join(root, file)
            # print(file_path)
            if file_path.endswith(".pth"):
                file_list.append(file_path)
    sample_list = random.sample(file_list, num)
    return sample_list


def random_change_model(model_state_dict):
    new_parameter_value = copy.deepcopy(model_state_dict)
    weight_keys = [key for key in model_state_dict.keys() if 'conv' in key and key.endswith('.weight')]
    random_key = random.choice(weight_keys)  # tamper conv layers
    # model_dict = model_state_dict[f'fc.weight']  # tamper fc layers
    model_dict = model_state_dict[f'{random_key}']
    total_params = model_dict.numel()
    num_params_to_modify = round(total_params * 0.01)  # tamper rate
    shape = model_dict.size()
    flattened_tensor = copy.deepcopy(model_dict.view(-1))
    indices = random.sample(range(total_params), num_params_to_modify)
    min = model_dict.min().detach().cpu().numpy()
    max = model_dict.max().detach().cpu().numpy()
    for index111 in indices:
        flattened_tensor[index111] = random.uniform(min, max)
    modified_tensor = flattened_tensor.view(shape)
    # new_parameter_value[f'fc.weight'] = modified_tensor
    new_parameter_value[f'{random_key}'] = modified_tensor
    return new_parameter_value


def test_model_change(orimodel, changemodel, sample_list):
    i = 0
    for file_path in sample_list:
        img = torch.load(file_path)
        orimodel.eval()
        changemodel.eval()
        output, _, _ = orimodel(img)
        gt = torch.argmax(output)
        gt = gt.unsqueeze(0)
        changemodel = changemodel.to(device)
        changeoutput, _, _ = changemodel(img)
        changegt = torch.argmax(changeoutput)
        changegt = changegt.unsqueeze(0).to(torch.device('cuda'))
        i += 1

        if gt != changegt:
            return True, i
    return False, i


model = clmgnet.CLMGNet(num_classes=16, n_bands=200, ps=13, inplanes=256, num_blocks=4, num_heads=4, num_encoders=1)
changemodel = clmgnet.CLMGNet(num_classes=16, n_bands=200, ps=13, inplanes=256, num_blocks=4, num_heads=4,
                              num_encoders=1)
model_state_dict = torch.load('./targetmodel.pth')
model_state_dict = model_state_dict['state_dict']
model.load_state_dict(model_state_dict)
device = torch.device('cuda')
model = model.to(device)
for k in range(0, 10):
    print(k)
    can = 0
    num = 0
    for j in range(0, 10000):
        new_cover_model_state_dict = random_change_model(model_state_dict)
        changemodel.load_state_dict(new_cover_model_state_dict)
        changemodel = changemodel.to(device)
        sample_list = random_select_sample("./samples", 5)
        cantest, num_test = test_model_change(model, changemodel, sample_list)
        if cantest:
            can += 1
            num += num_test
    with open(f"./a.text", "a") as file:
        # 将结果说明和结果写入文件中
        file.write("\n")
        file.write(f"{k}" + "        ")
        file.write("can detect：")
        file.write(str(can))
        file.write("used sample：")
        file.write(str(num))
        if can != 0:
            file.write("average：")
            file.write(str(num / can))
