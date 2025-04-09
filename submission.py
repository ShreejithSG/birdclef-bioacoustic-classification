from mlc.birdclef import ScorableModelTemplate
import librosa

class ScorableModel(ScorableModelTemplate):

    def predict(self, raw_files: list[str]):
        """Input argument will vary. See you competition's template.

        :param raw_files: list of file path strings, depends on competition
        :return predictions: dataframe or np.array, depends on competition
        """
        # Implement this: may return random predictions for the first assignment
        class_names = get_class_names()
        predictions = []

        for file_path in raw_files:
            y, sr = librosa.load(file_path, sr=None)
            segment_length = 5 * sr
            file_stem = Path(file_path).stem
            segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]

            for i, _ in enumerate(segments):
                row_id = f"{file_stem}_{i * 5}"
                probs = np.random.rand(len(class_names))
                row = {"row_id": row_id}
                row.update(dict(zip(class_names, probs)))
                predictions.append(row)

        return pd.DataFrame(predictions)

    def process_inputs(self, raw_files: list[str]):
        """Input argument will vary. See you competition's template.

        :param raw_files: list of file path strings, depends on competition
        :return: anything needed for you model to make predictions, e.g. features or processed data
        """
        # Implement this: only need to read in files for first assignment
        features = []
        for file_path in raw_files:
            y, sr = librosa.load(file_path, sr=None)
            segment_length = 5 * sr
            segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]
            features.append((file_path, segments))
        return features
        
# Intialize, runs: __check_rep__ to validate class
model = ScorableModel() # error will be raised if the above is not implemented correctly
