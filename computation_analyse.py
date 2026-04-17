import pandas as pd
import pickle as pkl


OU_size = 8


def logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size):
    total_calculation_number = 0
    actual_calculation_number = 0

    for i in range(0, len(layer_in_channel)):
        layer_total_calculation_number = (layer_in_channel[i] * kernel_size[i] * kernel_size[i] / OU_size) * layer_out_channel[i] * (feature_map_size[i] * feature_map_size[i])
        layer_actual_calculation_number = (layer_in_channel[i] * pattern_value_number[i] / OU_size) * (layer_out_channel[i] * best_keep_ratio[i]) * (feature_map_size[i] * feature_map_size[i])
        print(weight_name[i] + ': ' + str(layer_actual_calculation_number / layer_total_calculation_number))
        print(layer_total_calculation_number)

        total_calculation_number = total_calculation_number + layer_total_calculation_number
        actual_calculation_number = actual_calculation_number + layer_actual_calculation_number

    print('logical_calculation_percentage: ' + str(actual_calculation_number / total_calculation_number))
    print('logical_improvement: ' + str(1 / (actual_calculation_number / total_calculation_number)))

    if best_keep_ratio[1] != 1.0:
        return 1 / (actual_calculation_number / total_calculation_number).item()
    else:
        return 1 / (actual_calculation_number / total_calculation_number)


def weight_reducing_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, OU_size):
    total_calculation_number = 0
    actual_calculation_number = 0

    for i in range(0, len(layer_in_channel)):
        layer_total_weight_number = (layer_in_channel[i] * kernel_size[i] * kernel_size[i] / OU_size) * layer_out_channel[i]
        layer_actual_weight_number = (layer_in_channel[i] * pattern_value_number[i] / OU_size) * (layer_out_channel[i] * best_keep_ratio[i])
        print(weight_name[i] + ': ' + str(layer_actual_weight_number / layer_total_weight_number))
        print(layer_total_weight_number)

        total_calculation_number = total_calculation_number + layer_total_weight_number
        actual_calculation_number = actual_calculation_number + layer_actual_weight_number

    print('logical_calculation_percentage: ' + str(actual_calculation_number / total_calculation_number))
    print('logical_improvement: ' + str(1 / (actual_calculation_number / total_calculation_number)))

    if best_keep_ratio[1] != 1.0:
        return 1 / (actual_calculation_number / total_calculation_number).item()
    else:
        return 1 / (actual_calculation_number / total_calculation_number)


def get_model_logical_improvement(model_name, translate_name):
    if model_name == 'Vgg16':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                       'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                       'fc1.weight', 'fc2.weight', 'fc3.weight']
        feature_map_size = [32, 32, 16, 16, 8, 8, 8,
                            4, 4, 4, 2, 2, 2,
                            1, 1, 1]
        kernel_size = [3, 3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3,
                       1, 1, 1]
        layer_in_channel = [3, 64, 64, 128, 128, 256, 256,
                            256, 512, 512, 512, 512, 512,
                            512, 4096, 4096]
        layer_out_channel = [64, 64, 128, 128, 256, 256, 256,
                             512, 512, 512, 512, 512, 512,
                             4096, 4096, 10]
        pattern_value_number = [9, 9, 9, 9, 9, 9, 9,
                                9, 9, 9, 9, 9, 9,
                                1, 1, 1]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0]

        if translate_name == 'structure':
            pattern_value_number = [9, 9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65,
                                    9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65,
                                    0.25, 0.25, 1]

        elif translate_name == 'ORC':
            pattern_value_number = [9, 9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2,
                                    9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2,
                                    0.05, 0.05, 1.0]

        elif translate_name == 'weight_pattern_shape_translate':
            pattern_value_number = [8, 8, 8, 4, 4, 4, 4,
                                    4, 2, 2, 2, 2, 2,
                                    0.25, 0.25, 1]

        elif translate_name == 'weight_pattern_value_original_translate':
            with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'weight_pattern_value_normalized_translate':
            with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'structure_and_weight_pattern_value_original_translate':
            pattern_value_number = [9, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    0.25, 0.25, 1]
            with open('model_' + model_name + '_structure_and_value_original_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        else:
            pattern_value_number = [8, 8, 8, 4, 4, 4, 4,
                                    4, 2, 2, 2, 2, 2,
                                    0.25, 0.25, 1]
            with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        logical_improvement = logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)
        weight_reducing = weight_reducing_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, OU_size)

        return logical_improvement, weight_reducing


    if model_name == 'Res18':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight',
                       'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight',
                       'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                       'fc.weight']
        feature_map_size = [32, 32, 32, 32, 32, 16,
                            16, 16, 16, 8, 8, 8,
                            8, 4, 4, 4, 4,
                            16, 8, 4,
                            1]
        kernel_size = [3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3,
                       1, 1, 1,
                       1]
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
        pattern_value_number = [9, 9, 9, 9, 9, 9,
                                9, 9, 9, 9, 9, 9,
                                9, 9, 9, 9, 9,
                                1, 1, 1,
                                1]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0]

        if translate_name == 'structure':
            pattern_value_number = [9, 9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65,
                                    9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65,
                                    9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65,
                                    0.65, 0.65, 0.65,
                                    1.0]

        elif translate_name == 'ORC':
            pattern_value_number = [9, 9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2,
                                    9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2,
                                    9*0.2, 9*0.2, 9*0.2, 9*0.2, 9*0.2,
                                    0.2, 0.2, 0.2,
                                    1.0]

        elif translate_name == 'weight_pattern_shape_translate':
            pattern_value_number = [8, 4, 4, 4, 4, 4,
                                    4, 4, 4, 4, 4, 4,
                                    4, 4, 2, 2, 2,
                                    1, 1, 1,
                                    1]

        elif translate_name == 'weight_pattern_value_original_translate':
            with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'weight_pattern_value_normalized_translate':
            with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'structure_and_weight_pattern_value_original_translate':
            pattern_value_number = [9, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    0.65, 0.65, 0.65,
                                    1.0]
            with open('model_' + model_name + '_structure_and_value_original_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        else:
            pattern_value_number = [8, 4, 4, 4, 4, 4,
                                    4, 4, 4, 4, 4, 4,
                                    4, 4, 2, 2, 2,
                                    1, 1, 1,
                                    1]
            with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        logical_improvement = logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)
        weight_reducing = weight_reducing_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, OU_size)

        return logical_improvement, weight_reducing


    if model_name == 'Res50':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight',
                       'conv11.weight', 'conv12.weight', 'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight', 'conv18.weight', 'conv19.weight',
                       'conv20.weight', 'conv21.weight', 'conv22.weight', 'conv23.weight', 'conv24.weight', 'conv25.weight', 'conv26.weight', 'conv27.weight', 'conv28.weight',
                       'conv29.weight', 'conv30.weight', 'conv31.weight', 'conv32.weight', 'conv33.weight', 'conv34.weight', 'conv35.weight', 'conv36.weight', 'conv37.weight',
                       'conv38.weight', 'conv39.weight', 'conv40.weight', 'conv41.weight', 'conv42.weight', 'conv43.weight', 'conv44.weight', 'conv45.weight', 'conv46.weight',
                       'conv47.weight', 'conv48.weight', 'conv49.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight', 'shortcut4.weight',
                       'fc.weight']
        feature_map_size = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                            32, 16, 16, 16, 16, 16, 16, 16, 16,
                            16, 16, 16, 16, 8, 8, 8, 8, 8,
                            8, 8, 8, 8, 8, 8, 8, 8, 8,
                            8, 8, 8, 8, 4, 4, 4, 4, 4,
                            4, 4, 4,
                            32, 16, 8, 4,
                            1]
        kernel_size = [3, 1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1,
                       1, 1, 1, 1,
                       1]
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
        pattern_value_number = [9, 1, 9, 1, 1, 9, 1, 1, 9, 1,
                                1, 9, 1, 1, 9, 1, 1, 9, 1,
                                1, 9, 1, 1, 9, 1, 1, 9, 1,
                                1, 9, 1, 1, 9, 1, 1, 9, 1,
                                1, 9, 1, 1, 9, 1, 1, 9, 1,
                                1, 9, 1,
                                1, 1, 1, 1,
                                1]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0,
                           1.0]

        if translate_name == 'structure':
            pattern_value_number = [9, 0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75,
                                    0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75,
                                    0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75,
                                    0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75,
                                    0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75, 0.75, 9*0.65, 0.75,
                                    0.75, 9*0.65, 0.75,
                                    0.75, 0.75, 0.75, 0.75,
                                    1.0]

        elif translate_name == 'ORC':
            pattern_value_number = [9, 0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2,
                                    0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2,
                                    0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2,
                                    0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2,
                                    0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2, 0.2, 9*0.2, 0.2,
                                    0.2, 9*0.2, 0.2,
                                    0.2, 0.2, 0.2, 0.2,
                                    1.0]

        elif translate_name == 'weight_pattern_shape_translate':
            pattern_value_number = [8, 1, 8, 1, 1, 8, 1, 1, 8, 1,
                                    1, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                    0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                    0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                    0.5, 4, 1, 0.5, 2, 0.5, 0.5, 2, 0.5,
                                    0.5, 2, 0.5,
                                    1, 1, 1, 1,
                                    1]

        elif translate_name == 'weight_pattern_value_original_translate':
            with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'weight_pattern_value_normalized_translate':
            with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'structure_and_weight_pattern_value_original_translate':
            pattern_value_number = [9, 0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75,
                                    0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75,
                                    0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75,
                                    0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75,
                                    0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75, 0.75, 9 * 0.65, 0.75,
                                    0.75, 9 * 0.65, 0.75,
                                    0.75, 0.75, 0.75, 0.75,
                                    1.0]
            with open('model_' + model_name + '_structure_and_value_original_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        else:
            pattern_value_number = [8, 1, 8, 1, 1, 8, 1, 1, 8, 1,
                                    1, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                    0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                    0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                    0.5, 4, 1, 0.5, 2, 0.5, 0.5, 2, 0.5,
                                    0.5, 2, 0.5,
                                    1, 1, 1, 1,
                                    1]
            with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        logical_improvement = logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)
        weight_reducing = weight_reducing_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, OU_size)

        return logical_improvement, weight_reducing


    if model_name == 'WRN':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                       'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                       'fc.weight']
        feature_map_size = [32, 32, 32, 32, 32, 16, 16,
                            16, 16, 8, 8, 8, 8,
                            32, 16, 8,
                            1]
        kernel_size = [3, 3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3,
                       1, 1, 1,
                       1]
        layer_in_channel = [3, 16, 128, 128, 128, 128, 256,
                            256, 256, 256, 512, 512, 512,
                            16, 128, 256,
                            512]
        layer_out_channel = [16, 128, 128, 128, 128, 256, 256,
                             256, 256, 512, 512, 512, 512,
                             128, 256, 512,
                             10]
        pattern_value_number = [9, 9, 9, 9, 9, 9, 9,
                                9, 9, 9, 9, 9, 9,
                                1, 1, 1,
                                1]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0]

        if translate_name == 'structure':
            pattern_value_number = [9, 9, 9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65,
                                    9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65, 9*0.65,
                                    9*0.65, 9*0.65, 9*0.65,
                                    1.0]

        elif translate_name == 'ORC':
            pattern_value_number = [9, 9*0.15, 9*0.15, 9*0.15, 9*0.15, 9*0.15, 9*0.15,
                                    9*0.15, 9*0.15, 9*0.15, 9*0.15, 9*0.15, 9*0.15,
                                    0.15, 0.15, 0.15,
                                    1.0]

        elif translate_name == 'weight_pattern_shape_translate':
            pattern_value_number = [8, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 2, 2, 2,
                                    1, 1, 1,
                                    1]

        elif translate_name == 'weight_pattern_value_original_translate':
            with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'weight_pattern_value_normalized_translate':
            with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        elif translate_name == 'structure_and_weight_pattern_value_original_translate':
            pattern_value_number = [9, 9, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    9 * 0.65, 9 * 0.65, 9 * 0.65,
                                    1.0]
            with open('model_' + model_name + '_structure_and_value_original_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        else:
            pattern_value_number = [8, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 2, 2, 2,
                                    1, 1, 1,
                                    1]
            with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                reuse_ratio_information = pkl.load(f)
                f.close()
            for i in range(0, len(weight_name)):
                best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
                print(best_keep_ratio[i])

        logical_improvement = logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)
        weight_reducing = weight_reducing_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, OU_size)

        return logical_improvement, weight_reducing


if __name__ == '__main__':
    model_name = ['Vgg16', 'Res18', 'Res50', 'WRN']
    translate_name = ['structure_pruning', 'ORC_pruning', 'weight_pattern_shape_translate', 'weight_pattern_value_identical_translate', 'weight_pattern_value_similar_translate', 'structure_pruning_and_weight_pattern_value_identical_translate', 'weight_pattern_shape_and_value_similar_translate']

    result_computation = pd.DataFrame()
    result_weight = pd.DataFrame()
    logical_improvement = [0.0] * len(translate_name)
    weight_reducing = [0.0] * len(translate_name)
    cut_ratio_computation = [0.0] * len(translate_name)
    cut_ratio_weight = [0.0] * len(translate_name)
    for i in range(0, len(model_name)):
        for j in range(0, len(translate_name)):
            logical_improvement[j], weight_reducing[j] = get_model_logical_improvement(model_name[i], translate_name[j])
            cut_ratio_computation[j] = 1 - 1 / logical_improvement[j]
            cut_ratio_weight[j] = 1 - 1 / weight_reducing[j]

        result_computation[model_name[i] + '_logical_improvement'] = logical_improvement
        result_computation[model_name[i] + '_cut_ratio'] = cut_ratio_computation
        result_weight[model_name[i] + '_logical_improvement'] = weight_reducing
        result_weight[model_name[i] + '_cut_ratio'] = cut_ratio_weight

    result_computation.to_csv('logical_improvement_info' + '.csv')
    result_weight.to_csv('weight_reducing_info' + '.csv')
