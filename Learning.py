from Configuration import Configuration
from DataImporter import DataImporter

class Learning:

    @staticmethod
    def simple_binary_classification(self, data_set):

        di = DataImporter()

        data_set = di.import_all_data(data_description_file_path, standardized_photos_file_path)

        inputs = Input(shape=data_set["training"][0].shape[1:])
        layer_1 = Conv2D(32, 5, padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        layer_2 = MaxPooling2D(pool_size=(2, 2))(layer_1)
        layer_3 = Conv2D(32, 5, border_mode='valid', activation='relu', kernel_initializer='glorot_normal')(layer_2)
        layer_4 = MaxPooling2D(pool_size=(2, 2))(layer_3)
        layer_5 = Flatten()(layer_4)
        outputs = Dense(units=1, activation='sigmoid', name='prediction', kernel_initializer='glorot_normal')(layer_5)

        model = Model(inputs=inputs, outputs=[outputs])

        model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_acc',
                                       patience=0,
                                       verbose=0, mode='auto')

        results = model.fit(ds["training"][0],
                            ds["training"][1],
                            validation_data=ds["validation"],
                            epochs=10,
                            batch_size=320,
                            callbacks=[early_stopping])

        validation_accuracy = results.history["val_acc"][-1]

        evaluation = model.evaluate(ds["test"][0], ds["test"][1])

        return {
            "validation": validation_accuracy,
            "test": evaluation
        }

    def categorical_classification(self):
        pass

    def crow_score_regression(self):
        pass