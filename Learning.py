from DataImporter import DataImporter
import keras
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add
import os
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard as TensorBoardCallback
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import datetime
import gc


class Learning:

    @staticmethod
    def run_experiment(config):
        # bayesian reward primarily best performance,
        # if performance close, time cost for tie breaker
        # expand with one layer

        l = config["learning"]

        data_sets = Learning.get_data_sets(config)

        network_training_set = [data_sets["training"][0], data_sets["to_network"](data_sets["training"][1])]
        network_validation_set = [data_sets["validation"][0], data_sets["to_network"](data_sets["validation"][1])]

        input_shape = data_sets["training"][0].shape[1:]

        input = Input(shape=input_shape)

        label_count = data_sets["to_network"](data_sets["training"][1]).shape[1]

        model = Learning.build_model(input=input, label_count=label_count, output_type=data_sets["output_type"], **l)

        results = \
            Learning.train(
                model,
                l["target"]["output_type"],
                network_training_set,
                network_validation_set,
                l["hardware_limits"]["epochs"],
                l["hardware_limits"]["batch_size"],
                os.path.join(config["tensorboard_path"], config['configuration_name'])
                )
        stats = Learning.evaluate(model, results, data_sets, model.metrics_names, l["hardware_limits"]["batch_size"])

        if data_sets["output_type"] == 'categorical_int' or data_sets["output_type"] == 'bool':
            raw_performance = stats['stats']['validation']['metrics']['acc']
        else:
            raw_performance = -stats['stats']['validation']['metrics']['mean_squared_error']
        return raw_performance, stats

    @staticmethod
    def build_model(
            input,
            label_count,
            block_depth,
            output_type,
            blocks_per_rescale,
            stride_difference_per_rescale,
            filter_increase_factor_per_rescale,
            skip_distance,
            initial_stride,
            initial_filter_count,
            initial_kernel_size,
            l1_factor,
            l2_factor,
            **unused):

        weight_regularizer = keras.regularizers.l1_l2(l1=l1_factor, l2=l2_factor)

        initial_convolution = Learning.initial_convolution(input, initial_stride, initial_filter_count, initial_kernel_size, weight_regularizer)

        # depth = 0
        # while True:
        architecture = Learning.build_linear_architecture(
            initial_convolution,
            block_depth,
            blocks_per_rescale,
            skip_distance,
            stride_difference_per_rescale,
            filter_increase_factor_per_rescale,
            regularizer=weight_regularizer)

        raw_output = Learning.resnet_output(architecture, label_count, regularizer=weight_regularizer)

        if output_type == 'bool' or output_type == 'float':
            output = Activation('sigmoid')(raw_output)

        if output_type == 'categorical_int':
            output = Activation('softmax')(raw_output)

            # depth = depth + 1

            # time_cost = Learning.compute_time_cost(input, output, batch_size)
            #
            # if time_cost >= time_budget_per_batch: break

        return Model(inputs=input, outputs=output)

    # @staticmethod
    # def first_layer(inp, computation_budget, regularizer):
    #
    #     kernel_size = (3, 3)
    #     filter_count = computation_budget / (np.prod(inp.shape[1:] * np.prod(kernel_size)))
    #
    #     convoluted_weights = Conv2D(
    #         filter_count,
    #         (3, 3),
    #         stride=(1, 1),
    #         padding='same',
    #         kernel_initializer='glorot_normal',
    #         bias_initializer='glorot_normal',
    #         data_format="channels_last",
    #         kernel_regularizer=regularizer,
    #         bias_regularizer=regularizer)(inp)
    #
    #     normalized_weights = BatchNormalization(axis=3)(convoluted_weights)
    #
    #     return Activation('relu')(normalized_weights)

    @staticmethod
    def get_data_sets(import_config):
        di = DataImporter()

        # standardized_photo_path = di.construct_standardized_photo_path(import_config["data_import"]["standardized_photos"], import_config["data_import"]["output_image_dimensions"])

        data_set_dictionary = di.import_all_data(import_config["data_import"]["data_description_file_path"],
                                                 os.path.join(import_config["data_import"]["standardized_photos"], 'output_image_dimensions@' + json.dumps(import_config["data_import"]["output_image_dimensions"])))

        target_config = import_config["learning"]["target"]

        data_set_dictionary["output_type"] = target_config["output_type"]

        if isinstance(target_config["value_names"], list):
            data_set_dictionary["value_names"] = target_config["value_names"]
        else:

            data_set_dictionary["value_names"] = dict(zip(
                data_set_dictionary["training"][1][target_config["column_header"]],
                data_set_dictionary["meta"]["training"][target_config["value_names"]]
            ))

        data_set_dictionary["training"] = (data_set_dictionary["training"][0], np.array(data_set_dictionary["training"][1][target_config["column_header"]]))
        data_set_dictionary["validation"] = (data_set_dictionary["validation"][0], np.array(data_set_dictionary["validation"][1][target_config["column_header"]]))
        data_set_dictionary["test"] = (data_set_dictionary["test"][0], np.array(data_set_dictionary["test"][1][target_config["column_header"]]))

        if target_config["output_type"] == 'bool':
            data_set_dictionary["to_network"] = lambda x: np.reshape(x, [-1, 1])
            data_set_dictionary["from_network"] = lambda x: np.reshape(x, [-1])

        if target_config["output_type"] == 'float':
            scale = np.max(data_set_dictionary["training"][1])
            data_set_dictionary["to_network"] = lambda x: np.reshape(x, [-1, 1]) / scale
            data_set_dictionary["from_network"] = lambda x: np.reshape(x, [-1]) * scale

        if target_config["output_type"] == 'categorical_int':
            scale = np.max(data_set_dictionary["training"][1])
            data_set_dictionary["to_network"] = lambda x: to_categorical(x)
            data_set_dictionary["from_network"] = lambda x: np.argmax(x, axis=-1)

        return data_set_dictionary

    @staticmethod
    def train(model, output_type, training_set, validation_set, epochs, batch_size, tensorboard_path):

        if output_type == 'bool':
            loss = 'binary_crossentropy'
            metrics = ['accuracy']

        if output_type == 'categorical_int':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        if output_type == 'float':
            loss = 'mean_squared_error'
            metrics = ['mse']

        model.compile(optimizer='adam', loss=loss, metrics=metrics)

        # reduce learning rate when stagnating
        validation_based_learning_rate = ReduceLROnPlateau(patience=1, factor=.5, mode='min', min_lr=0)

        # and when that has not worked for a few epochs, just stop
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3,
                                       verbose=0, mode='auto')

        tensorboard_logger = TensorBoardCallback(log_dir=tensorboard_path)

        results = model.fit(training_set[0],
                            training_set[1],
                            validation_data=validation_set,
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            callbacks=[validation_based_learning_rate, tensorboard_logger, early_stopping])

        return results

    @staticmethod
    def evaluate(model, results, data_set, metric_names, batch_size):
        training_metrics = model.evaluate(data_set["training"][0], data_set["to_network"](data_set["training"][1]))
        training_prediction = data_set["from_network"](model.predict(data_set["training"][0], batch_size=batch_size))

        validation_metrics = model.evaluate(data_set["validation"][0], data_set["to_network"](data_set["validation"][1]))
        validation_prediction = data_set["from_network"](model.predict(data_set["validation"][0], batch_size=batch_size))

        test_metrics = model.evaluate(data_set["test"][0], data_set["to_network"](data_set["test"][1]))
        test_prediction = data_set["from_network"](model.predict(data_set["test"][0], batch_size=batch_size))

        return {
            "stats": {
                "training": {
                    "history": results.history,
                    "metrics": dict(zip(metric_names, training_metrics)),
                    "correct": data_set["training"][1].tolist(),
                    "predicted": training_prediction.reshape(-1).tolist()
                },
                "validation": {
                    "metrics": dict(zip(metric_names, validation_metrics)),
                    "correct": data_set["validation"][1].tolist(),
                    "predicted": validation_prediction.reshape(-1).tolist()
                },
                "test": {
                    "metrics": dict(zip(metric_names, test_metrics)),
                    "correct": data_set["test"][1].tolist(),
                    "predicted": test_prediction.reshape(-1).tolist()
                },
                "meta": {
                    "type": data_set["output_type"],
                    "labels": data_set["value_names"]
                }
            },
            "model": model
        }

    @staticmethod
    def build_linear_architecture(
            input,
            block_depth,
            blocks_per_rescale,
            skip_distance,
            stride_difference_per_rescale,
            filter_increase_factor_per_rescale,
            regularizer):

        cursor = input

        residual_stride = 0
        # total_layer_cost = 0
        for i in range(int(block_depth)):
            stride = 1
            filter_increase_factor = 1

            if i % blocks_per_rescale == 0:

                stride = int(stride_difference_per_rescale + residual_stride)
                residual_stride = (stride_difference_per_rescale + residual_stride) % 1
                filter_increase_factor = filter_increase_factor_per_rescale

            next_skip_bock = Learning.resnet_skip_block(cursor,
                                                        stride,
                                                        filter_increase_factor,
                                                        skip_distance,
                                                        regularizer)

            # total_layer_cost += next_layer_cost

            cursor = Activation('relu')(next_skip_bock)

        output = cursor

        return output

    # @staticmethod
    # def build_tree_auto_architecture(input, layer_constructor, block_count, skip_distance, flops_scale, resolution_to_feature_ratio, regularizer):
    #
    #     skip_depth = int(np.log(block_count) / np.log(skip_distance))
    #
    #     def tree_node(inp, depth):
    #         if(depth == 0):
    #             return layer_constructor(inp, flops_scale, resolution_to_feature_ratio, regularizer)
    #
    #         return Learning.resnet_skip_block(inp, lambda inp: tree_node(inp, depth -1), skip_distance, flops_scale, resolution_to_feature_ratio, regularizer=regularizer)
    #
    #     output = tree_node(input, skip_depth)
    #
    #     return input, output

    @staticmethod
    def resnet_output(inp, label_count, regularizer):

        # By taking a pool size as large as the spatial dimensions of the input, the average of all filters is computed.
        # The network then still has two pointless spatial dimensions of size 1. Flatten removes those dimensions.
        pool_shape = tuple(inp.shape.as_list()[1:3])
        network_feature_scores = Flatten()(AveragePooling2D(pool_size=pool_shape)(inp))

        # All network feature outputs are now constructed, and can be aggregated into label_scores using a dense layer.
        # I really hate this Dense layer because it in all likelihood will consume about half of the computation budget...
        label_scores = Dense(units=int(label_count), kernel_initializer='glorot_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer)(network_feature_scores)

        # average_pooling_cost = np.prod(inp.shape[1:].as_list())
        # dense_cost = inp.shape[-1].value * label_count

        # cost = average_pooling_cost + dense_cost

        return label_scores  #, cost

    # @staticmethod
    # def dense(inp, label_count):
    #     label_scores = Dense(units=label_count, kernel_initializer='glorot_normal', kernel_regularizer=None, bias_regularizer=None)(inp)
    #
    #     return label_scores, inp.shape[-1].value * label_count
    #
    # @staticmethod
    # def avg_pool(inp):
    #     label_scores = AveragePooling2D(pool_size=tuple(inp.shape[1:3].as_list()))(inp)
    #
    #     return label_scores, np.prod(inp.shape[1:3].as_list())

    # def traditional_output(self, inp, label_count, regularizer):
    #     return Dense(units=label_count, kernel_initializer='glorot_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer)(Flatten(inp))

    @staticmethod
    def resnet_skip_block(inp, stride, filter_increase_factor, skip_distance, regularizer):

        cost = 0

        current = inp

        for i in range(int(skip_distance)):

            next_stride = stride if i == 0 else 1
            next_filter_increase_factor = filter_increase_factor if i == 0 else 1

            current = Learning.resnet_convolution(current, stride=next_stride, filter_increase_factor=next_filter_increase_factor, regularizer=regularizer)
            # cost += cost_increase

            if i < skip_distance - 1:
                current = Activation("relu")(current)

        skip_output = inp
        if stride != 1 or filter_increase_factor != 1:

            skip_output = Learning.resnet_convolution(inp, stride=stride, filter_increase_factor=filter_increase_factor, regularizer=regularizer)
            # cost += cost_increase

        merged_with_bypass = Add()([skip_output, current])


        return merged_with_bypass  #, cost

    # @staticmethod
    # def scale_stride_and_filter_count(inp, flops_scale, resolution_to_feature_ratio):
    #     stride = (1 / flops_scale) / resolution_to_feature_ratio
    #     feature_scale = (1 / flops_scale) * resolution_to_feature_ratio
    #
    #     filter_count = inp.shape.as_list()[3] * feature_scale
    #
    #     return stride, filter_count

    @staticmethod
    def resnet_convolution(inp, stride, filter_increase_factor, regularizer):

        filter_count = int(round(inp.shape[-1].value * filter_increase_factor))

        kernel_size = (3, 3)

        convoluted_weights = Conv2D(
            filter_count,
            kernel_size,
            strides=(int(stride), int(stride)),
            padding='same',
            kernel_initializer='glorot_normal',
            bias_initializer='glorot_normal',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            data_format="channels_last")(inp)

        normalized_weights = BatchNormalization(axis=3)(convoluted_weights)

        # cost = np.prod(kernel_size) * np.prod(normalized_weights.shape[1:].as_list()) * inp.shape[-1].value

        return normalized_weights  #, cost

    @staticmethod
    def initial_convolution(inp, initial_stride, initial_filter_count, initial_kernel_size, regularizer):

        kernel_size = (int(initial_kernel_size), int(initial_kernel_size))

        convoluted_weights = Conv2D(
            int(initial_filter_count),
            kernel_size,
            strides=(int(initial_stride), int(initial_stride)),
            padding='same',
            kernel_initializer='glorot_normal',
            bias_initializer='glorot_normal',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            data_format="channels_last")(inp)

        normalized_weights = BatchNormalization(axis=3)(convoluted_weights)

        return Activation('relu')(normalized_weights)

    # @staticmethod
    # def inception_layer(inp, **base_layer_config):
    #     convolutional_tower = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_normal', **base_layer_config)(inp)
    #     convolutional_tower = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal', **base_layer_config)(convolutional_tower)
    #     pooling_tower = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inp)
    #     pooling_tower = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_normal', **base_layer_config)(pooling_tower)
    #
    #     return keras.layers.concatenate([convolutional_tower, pooling_tower], axis=3)

    @staticmethod
    def compute_time_cost(input, output, batch_size):

        input_shape = input.shape.as_list()

        inp = np.random.random([batch_size] + input_shape[1:])

        result_list = []

        repetitions = 25
        # gc.collect()
        # gc.disable()
        a = datetime.datetime.now()

        for i in range(repetitions):
            x = output.eval({input: inp}, K.get_session())
            result_list.append(x)

        b = datetime.datetime.now()
        time_cost = b - a
        # gc.enable()
        # gc.collect()

        as_array = np.array(result_list)
        print("stats")
        print(np.mean(as_array))
        print(np.std(as_array))

        return (time_cost.microseconds / repetitions) / 1000.0