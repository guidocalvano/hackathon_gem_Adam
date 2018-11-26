import os
import os.path
import json
import sklearn
import sklearn.metrics
from keras.models import load_model
import scipy
import scipy.stats
import pandas as pd


class Analysis:

    @staticmethod
    def store_raw_result(result_path, result):
        raw_path = os.path.join(result_path, 'raw')
        model_file_path = os.path.join(raw_path, 'model.h5')
        stats_file_path = os.path.join(raw_path,  'stats.json')

        model = result["model"]
        stats = result["stats"]

        os.makedirs(raw_path, exist_ok=True)
        model.save(model_file_path)

        with open(stats_file_path, 'w') as f:
            json.dump(stats, f)

    @staticmethod
    def load_raw_result(report_path):
        raw_path = os.path.join(report_path, 'raw')
        model_file_path = os.path.join(raw_path, 'model.h5')
        stats_file_path = os.path.join(raw_path,  'stats.json')

        model = load_model(model_file_path)
        with open(stats_file_path) as f:
            stats = json.load(f)

        return {
            "model": model,
            "stats": stats
        }

    @staticmethod
    def process_result(result_path):

        result = Analysis.load_raw_result(result_path)

        statistics = Analysis.compute_statistics(result)

        Analysis.produce_report(result_path, statistics)

    @staticmethod
    def compute_statistics(result):
        resultType = result["stats"]["meta"]["type"]
        if resultType == "binary" or resultType == "categorical":
            return Analysis.compute_classification_statistics(result["stats"])

        if resultType == "regression":
            return Analysis.compute_regression_statistics(result["stats"])

    @staticmethod
    def compute_classification_statistics(result):

        training_report = sklearn.metrics.classification_report(
            result["training"]["correct"],
            result["training"]["predicted"],
            target_names=result["meta"]["labels"])
        validation_report = sklearn.metrics.classification_report(
            result["validation"]["correct"],
            result["validation"]["predicted"],
            target_names=result["meta"]["labels"])
        test_report = sklearn.metrics.classification_report(
            result["test"]["correct"],
            result["test"]["predicted"],
            target_names=result["meta"]["labels"])

        training_confusion = sklearn.metrics.confusion_matrix(
            result["training"]["correct"],
            result["training"]["predicted"])
        validation_confusion = sklearn.metrics.confusion_matrix(
            result["validation"]["correct"],
            result["validation"]["predicted"])
        test_confusion = sklearn.metrics.confusion_matrix(
            result["test"]["correct"],
            result["test"]["predicted"])

        return {
            "meta": {
                "type": "categorical",
                "labels": result["meta"]["labels"]
            },
            "training": {
                "classification": training_report,
                "confusion": training_confusion,
                "history": result["training"]
            },
            "validation": {
                "classification": validation_report,
                "confusion": validation_confusion,
                "metrics": result["validation"]["metrics"]
            },
            "test": {
                "classification": test_report,
                "confusion": test_confusion,
                "metrics": result["test"]["metrics"]
            }
        }

    @staticmethod
    def compute_regression_stats_for_predictions(y_true, y_pred):

        pearson_r, p_value = scipy.stats.pearsonr(y_true, y_pred)

        return {
                "r2": sklearn.metrics.r2_score(y_true, y_pred),
                "pearson_r": pearson_r,
                "pearson_p_value": p_value,
                "explained_variance_score": sklearn.metrics.explained_variance_score(y_true, y_pred),
                "mean_squared_error": sklearn.metrics.mean_squared_error(y_true, y_pred),
                "mean_absolute_error": sklearn.metrics.mean_absolute_error(y_true, y_pred),

            }

    @staticmethod
    def compute_regression_statistics(result):

        training_stats = Analysis.compute_regression_stats_for_predictions(result["training"]["correct"], result["training"]["predicted"])
        validation_stats = Analysis.compute_regression_stats_for_predictions(result["validation"]["correct"], result["validation"]["predicted"])
        test_stats = Analysis.compute_regression_stats_for_predictions(result["test"]["correct"], result["test"]["predicted"])

        return {
            "meta": {
                "type": "regression"
            },
            "training": {
                "history": result["training"]["history"],
                "regression": training_stats
            },
            "validation": {
                "metrics": result["validation"]["metrics"],
                "regression": validation_stats
            },
            "test": {
                "metrics": result["test"]["metrics"],
                "regression": test_stats
            }
        }

    @staticmethod
    def produce_report(result_path, statistics):
        report_path = os.path.join(result_path, 'report')
        os.makedirs(report_path, exist_ok=True)

        if statistics["meta"]["type"] == "categorical":
            return Analysis.produce_categorical_report(report_path, statistics)

        if statistics["meta"]["type"] == "regression":
            return Analysis.produce_regression_report(report_path, statistics)


    @staticmethod
    def produce_categorical_report(report_path, statistics):

        row_names = list(map(lambda v: v + '_true', statistics["meta"]["labels"]))
        column_names = list(map(lambda v: v + '_pred', statistics["meta"]["labels"]))

        training_report_path = os.path.join(report_path, 'training')
        validation_report_path = os.path.join(report_path, 'validation')
        test_report_path = os.path.join(report_path, 'test')

        os.makedirs(training_report_path, exist_ok=True)
        os.makedirs(validation_report_path, exist_ok=True)
        os.makedirs(test_report_path, exist_ok=True)


        Analysis.save_classification_report_csv(statistics["training"]["classification"], os.path.join(training_report_path, 'classification.csv'))

        pd.DataFrame(statistics["training"]["confusion"], columns=column_names, index=row_names).to_csv(
            os.path.join(training_report_path, 'confusion.csv'))


        Analysis.save_classification_report_csv(statistics["validation"]["classification"], os.path.join(validation_report_path, 'classification.csv'))

        pd.DataFrame(statistics["validation"]["confusion"], columns=column_names, index=row_names).to_csv(
            os.path.join(validation_report_path, 'confusion.csv'))


        Analysis.save_classification_report_csv(statistics["test"]["classification"], os.path.join(test_report_path, 'classification.csv'))

        pd.DataFrame(statistics["test"]["confusion"], columns=column_names, index=row_names).to_csv(
            os.path.join(test_report_path, 'confusion.csv'))


    @staticmethod
    def save_classification_report_csv(report, file_path):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.strip().split('      ')
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)

        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(file_path, index=False)


    @staticmethod
    def produce_regression_report(report_path, statistics):
        training_report_path = os.path.join(report_path, 'training')
        validation_report_path = os.path.join(report_path, 'validation')
        test_report_path = os.path.join(report_path, 'test')

        os.makedirs(training_report_path, exist_ok=True)
        os.makedirs(validation_report_path, exist_ok=True)
        os.makedirs(test_report_path, exist_ok=True)

        pd.DataFrame([statistics["training"]["regression"]]).to_csv(os.path.join(training_report_path, 'regression.csv'), index=False)
        with open(os.path.join(training_report_path, 'regression.json'), "w") as report_file:
            json.dump(statistics["training"]["regression"], report_file)

        pd.DataFrame([statistics["validation"]["regression"]]).to_csv(os.path.join(validation_report_path, 'regression.csv'), index=False)
        with open(os.path.join(validation_report_path, 'regression.json'), "w") as report_file:
            json.dump(statistics["validation"]["regression"], report_file)

        pd.DataFrame([statistics["test"]["regression"]]).to_csv(os.path.join(test_report_path, 'regression.csv'), index=False)
        with open(os.path.join(test_report_path, 'regression.json'), "w") as report_file:
            json.dump(statistics["test"]["regression"], report_file)