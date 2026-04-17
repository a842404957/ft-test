import pandas as pd
import pickle as pkl


def count_linear_relationship():
    result = pd.DataFrame()

    model_linear_relationship = [0] * 5
    with open('model_Vgg16_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
        multiple_relationship_information = pkl.load(f)
        f.close()
    with open('model_Vgg16_shape_and_value_normalized_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()

    layer_in_channel = [3, 64, 64, 128, 128, 256, 256,
                        256, 512, 512, 512, 512, 512,
                        512, 4096, 4096]
    layer_out_channel = [64, 64, 128, 128, 256, 256, 256,
                         512, 512, 512, 512, 512, 512,
                         4096, 4096, 10]
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                   'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                   'fc1.weight', 'fc2.weight', 'fc3.weight']

    for i in range(1, len(weight_name) - 1):
        print(weight_name[i])
        total_reuse_number = 0
        layer_linear_relationship = [0] * 5
        if 'conv' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i]):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1
        if 'fc' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i], 32):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1

        for j in range(0, 5):
            model_linear_relationship[j] = model_linear_relationship[j] + layer_linear_relationship[j] / total_reuse_number / (len(weight_name) - 2)

    result['Vgg16'] = model_linear_relationship


    model_linear_relationship = [0] * 5
    with open('model_Res18_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
        multiple_relationship_information = pkl.load(f)
        f.close()
    with open('model_Res18_shape_and_value_normalized_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()

    layer_in_channel = [3, 64, 64, 64, 64, 64,
                        128, 128, 128, 128, 256, 256,
                        256, 256, 512, 512, 512,
                        64, 128, 256,
                        512]
    layer_out_channel = [64, 64, 64, 64, 64, 128,
                         128, 128, 128, 256, 256, 256,
                         256, 512, 512, 512, 512,
                         128, 256, 512,
                         10]
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight',
                   'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight',
                   'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                   'fc.weight']

    for i in range(1, len(weight_name) - 1):
        print(weight_name[i])
        total_reuse_number = 0
        layer_linear_relationship = [0] * 5
        if 'conv' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i]):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1
        if 'shortcut' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i], 8):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1
        if 'fc' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i], 32):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1

        for j in range(0, 5):
            model_linear_relationship[j] = model_linear_relationship[j] + layer_linear_relationship[j] / total_reuse_number / (len(weight_name) - 2)

    result['Res18'] = model_linear_relationship


    model_linear_relationship = [0] * 5
    with open('model_Res50_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
        multiple_relationship_information = pkl.load(f)
        f.close()
    with open('model_Res50_shape_and_value_normalized_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()

    layer_in_channel = [3, 64, 64, 64, 256, 64, 64, 256, 64, 64,
                        256, 128, 128, 512, 128, 128, 512, 128, 128,
                        512, 128, 128, 512, 256, 256, 1024, 256, 256,
                        1024, 256, 256, 1024, 256, 256, 1024, 256, 256,
                        1024, 256, 256, 1024, 512, 512, 2048, 512, 512,
                        2048, 512, 512,
                        64, 256, 512, 1024,
                        2048]
    layer_out_channel = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256,
                         128, 128, 512, 128, 128, 512, 128, 128, 512,
                         128, 128, 512, 256, 256, 1024, 256, 256, 1024,
                         256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
                         256, 256, 1024, 512, 512, 2048, 512, 512, 2048,
                         512, 512, 2048,
                         256, 512, 1024, 2048,
                         10]
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight',
                   'conv11.weight', 'conv12.weight', 'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight', 'conv18.weight', 'conv19.weight',
                   'conv20.weight', 'conv21.weight', 'conv22.weight', 'conv23.weight', 'conv24.weight', 'conv25.weight', 'conv26.weight', 'conv27.weight', 'conv28.weight',
                   'conv29.weight', 'conv30.weight', 'conv31.weight', 'conv32.weight', 'conv33.weight', 'conv34.weight', 'conv35.weight', 'conv36.weight', 'conv37.weight',
                   'conv38.weight', 'conv39.weight', 'conv40.weight', 'conv41.weight', 'conv42.weight', 'conv43.weight', 'conv44.weight', 'conv45.weight', 'conv46.weight',
                   'conv47.weight', 'conv48.weight', 'conv49.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight', 'shortcut4.weight',
                   'fc.weight']

    for i in range(1, len(weight_name) - 1):
        print(weight_name[i])
        total_reuse_number = 0
        layer_linear_relationship = [0] * 5
        if 'conv' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i]):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1
        if 'shortcut' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i], 8):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1
        if 'fc' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i], 32):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1

        for j in range(0, 5):
            model_linear_relationship[j] = model_linear_relationship[j] + layer_linear_relationship[j] / total_reuse_number / (len(weight_name) - 2)

    result['Res50'] = model_linear_relationship


    model_linear_relationship = [0] * 5
    with open('model_WRN_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
        multiple_relationship_information = pkl.load(f)
        f.close()
    with open('model_WRN_shape_and_value_normalized_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()

    layer_in_channel = [3, 16, 128, 128, 128, 128, 256,
                        256, 256, 256, 512, 512, 512,
                        16, 128, 256,
                        512]
    layer_out_channel = [16, 128, 128, 128, 128, 256, 256,
                         256, 256, 512, 512, 512, 512,
                         128, 256, 512,
                         10]
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                   'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                   'fc.weight']

    for i in range(1, len(weight_name) - 1):
        print(weight_name[i])
        total_reuse_number = 0
        layer_linear_relationship = [0] * 5
        if 'conv' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i]):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1
        if 'shortcut' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i], 8):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in][0][0] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1
        if 'fc' in weight_name[i]:
            for c_in in range(0, layer_in_channel[i], 32):
                reuse_number = 0
                for c_out in range(0, layer_out_channel[i]):
                    reuse_number = reuse_number + 1
                    if map_information[weight_name[i]][c_in][c_out][0] == -1:
                        break
                total_reuse_number = total_reuse_number + reuse_number
                for c_out in range(0, reuse_number):
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 0:
                        layer_linear_relationship[0] = layer_linear_relationship[0] + 1
                    if multiple_relationship_information[weight_name[i]][c_out][c_in] == 1:
                        layer_linear_relationship[1] = layer_linear_relationship[1] + 1
                    if 0.25 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 1:
                        layer_linear_relationship[2] = layer_linear_relationship[2] + 1
                    if 0.0625 <= multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.25:
                        layer_linear_relationship[3] = layer_linear_relationship[3] + 1
                    if 0 < multiple_relationship_information[weight_name[i]][c_out][c_in] < 0.0625:
                        layer_linear_relationship[4] = layer_linear_relationship[4] + 1

        for j in range(0, 5):
            model_linear_relationship[j] = model_linear_relationship[j] + layer_linear_relationship[j] / total_reuse_number / (len(weight_name) - 2)

    result['WRN'] = model_linear_relationship

    result.to_csv('model_linear_relationship' + '.csv')


if __name__ == '__main__':
    count_linear_relationship()

