class Model:
    """
    Class to build, train and use models for classification. There are different types of models:
    ['random_forest', 'mlp', 'mlp_binary']

    ...

    Methods
    -------
    build:
        build the model depending on the model_type choosen.
    train: (data: data object)
        trains the model with the given data.
    evaluate: (data: data object)
        returns an evaluation of the model given a test data.
    predict: (x: iterable - np.array)
        returns the prediction for new data.
    save: (path: str)
        saves the model for later use.
    load: (path: str)
        loads a pretrained model.
    """

    def __init__(self, model_type='random_forest', name=None, data=None):
        """
        Builds the model type specified or loads if name is provided.
        The path will be set as models/<model_type>. The model will be saved at
        models/<model_type>/<name>, and a file that contains information needed to encode
        and decode variables will be saved at models/<model_type>/extras.pkl.

        :param model_type: decide which model to train from a set of models. (str)
                ['random_forest', 'mlp']    (default: random_forest)
        :param name: name of the file where the model will be saved with the proper extension.
            TODO: automatically handle extension based on model_type

        ...

        Attributes
        ----------
        possible_model: (list)
            List of all the possible models to be implemented.
        path: (str)
            Created based on the model type. 'checkpoints/<model_type>/<model_name>'
        encoder_classes: (np.array)
            Classes that are encoded into int. (Needed to encode and decode model output)
        uniques: (np.array)
            Int that represent classes. (Needed to decode model output)
        model_type: (str)
            Model type being implemented.
        model: (obj)
            Object of keras or sklearn with a model.
        """

        self.possible_model = ['random_forest', 'mlp', 'mlp_binary']

        if model_type not in self.possible_model:
            raise "You must specify a valid model name from: {}".format(self.possible_model)

        self.path = 'checkpoints/' + model_type + '/'

        self.encoder_classes = None
        self.uniques = None
        self.model_type = model_type

        if name is None:
            self.model = None
            self.build(data)
        else:
            self.load(name)

    def build(self, data):
        """
        Builds the classifier from a set of possible models.

        :param data: data object to obtain input/output dimensions. (Data object)
            TODO: be able to change parameters outside build function.
        """

        model = None
        if data is None:
            input_dim = 15
            output_dim = 2
        else:
            input_dim = data.x_train.shape[1]
            output_dim = len(np.unique(data.y_train))

        if output_dim < 2:
            raise Exception('There must be at least two classes for classification.')

        if self.model_type == 'random_forest':
            model = RandomForestClassifier(random_state=0, criterion='entropy', n_estimators=10, verbose=True)
        elif self.model_type == 'mlp':
            model = Sequential()
            model.add(Dense(128, input_dim=input_dim))
            model.add(Activation('relu'))
            model.add(Dense(output_dim))
            model.add(Activation('softmax'))
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            model.summary()
        elif self.model_type == 'mlp_binary':
            if output_dim > 2:
                raise Exception('For mlp_binary there must only be two classes, but got {} classes'.format(output_dim))

            model = Sequential()
            model.add(Dense(128, input_dim=input_dim))
            model.add(Activation('relu'))
            model.add(Dense(128))
            model.add(Activation('relu'))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.summary()

        if model is None:
            raise Exception("Something went wrong, check the build method")
        self.model = model


    def train(self, data, n_epochs=15):
        """
        Trains the model with the data object passed.

        :param n_epochs: number of epochs to train the models (if needed).
        :param data: object containing training and test data. (data object)
        """

        self.encoder_classes = data.encoder_classes
        if data.uniques is not None:
            self.uniques = data.uniques

        if self.model_type == 'random_forest':
            self.model.fit(data.x_train, data.y_train)
        if self.model_type == 'mlp':
            self.model.fit(data.x_train, data.y_train, epochs=n_epochs, batch_size=2)
        if self.model_type == 'mlp_binary':
            self.model.fit(data.x_train, data.y_train, epochs=n_epochs, batch_size=2)

    def evaluate(self, data):
        """
        Evaluates the model with the data object passed.
        :param data: object containing training and test data. (data object)
        :return: the evaluation scores # TODO: What are the evaluation scores?
        """

        if self.model_type == 'random_forest':
            return self.model.score(data.x_test, data.y_test)
        if self.model_type == 'mlp':
            return self.model.evaluate(data.x_test, data.y_test, batch_size=2)
        if self.model_type == 'mlp_binary':
            return self.model.evaluate(data.x_test, data.y_test, batch_size=2)

    def predict(self, x):
        """
        Predict the output of a model given an input

        :param x: features from a new user or group of users. (np.array)
        :return: model prediction for the given user.

        """

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        x = normalize_data(x)

        output = None

        if self.model_type == 'random_forest':
            output = self.model.predict(x)
        if self.model_type == 'mlp':
            output = self.model.predict(x)
        if self.model_type == 'mlp_binary':
            output = self.model.predict(x)

        if output is None:
            raise Exception('Something went wrong, check predict method')

        if self.encoder_classes is not None:
            output = get_labels(output, self.encoder_classes, self.uniques)

        return output

   def test_set(self, dataset):
        """
        Method used to test the model on a test dataset.
        :param dataset: labelled dataset with testing set. (dataset object)
        :return: mean accuracy of the dataset.
            # TODO: change the evaluation metric.
        """

        if 'label' not in dataset.df.columns:
            raise Exception('It must be a labelled dataset')

        columns = list(dataset.df.columns)
        columns.remove('user')
        columns.remove('label')

        x = dataset.get_data(columns)
        y = dataset.df.label

        pred = self.predict(x)

        total = len(y)
        y = [x.lower() for x in y]
        y = np.array(y)
        accuracy = np.sum(pred == y) / total

        print('\n Accuracy: {}'.format(accuracy))
        return accuracy

def save(self, name):
        """
        Saves the model to file for later use.
        The path should be 'checkpoints/<model_type>/<model_name>'
        :param name: name of the model to be saved with the proper extension. (str)
            # TODO: automatically handle extension based on model_type.
        """
        self.path += name.split('.')[0]

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        file_path = os.path.join(self.path, name)
        if self.model_type in ['mlp', 'mlp_binary']:
            self.model.save(file_path)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)

        extras_path = os.path.join(self.path, 'extras.pkl')
        extra = {'encoder_classes': self.encoder_classes, 'uniques': self.uniques}
        with open(extras_path, 'wb') as f:
            pickle.dump(extra, f)

    def load(self, name):
        """
        Loads a trained model for inference or further training.
        The path should be 'checkpoints/<model_type>/<model_name>'
        :param name: path and name of the model to load.
        """
        self.path += name.split('.')[0]

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        file_path = os.path.join(self.path, name)
        if not os.path.exists(file_path):
            raise Exception('There is no model in the specified path.')

        if self.model_type in ['mlp', 'mlp_binary']:
            self.model = load_model(file_path)
        else:
            self.model = pickle.load(open(file_path, 'rb'))

        extras_path = os.path.join(self.path, 'extras.pkl')
        with open(extras_path, 'rb') as f:
            extras = pickle.load(f)

        self.uniques = extras['uniques']
        self.encoder_classes = extras['encoder_classes']


    def predict_user(self, twitter_api, user):
        """
        Method to obtain the prediction for a given user name (screen_name)
        :param twitter_api: twitter api object already verified ready to be used. (twitter object)
        :param user: user screen_name, example: @Basetis -> Basetis. (str)
        :return: model prediction of user class. (str)
        """

        response = twitter_api.statuses.user_timeline(screen_name=user, tweet_mode='extended', count=10)

        row = get_user_features(response)
        x = np.array(row.values())

        output = self.predict(x)

        return output



class Data:
    """
    Class to handle splitting, normalizing, encoding ... data to input to the models.
    """

    def __init__(self, x, y, test_size=0.2, to_categorical=True):
        """
        Class to handle splitting, normalizing, encoding ... data to input to the models.

        :param x: users features (n_samples, n_features). (np.array)
        :param y: possible labels (n_samples). (np.array)
        :param test_size: percentage of samples for testing. (int from 0-1)
        """

        self.encoder_classes = None
        self.uniques = None

        x = normalize_data(x)
        y = self.encode_labels(y, to_categorical=to_categorical)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                                                test_size=test_size,
                                                                                random_state=42)

    def encode_labels(self, y, to_categorical):
        """
        Encodes the classes from y and if to_categorical, creates one-hot encoding for the variables.
        :param y: original labels, usually strings. (np.array)
        :param to_categorical: whether to do one-hot encoding. (bool)
        :return: encoded (or one-hot encoded) variables. (np.array)
        """
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        self.encoder_classes = encoder.classes_
        if to_categorical:
            uniques, ids = np.unique(encoded_y, return_inverse=True)
            self.uniques = uniques
            return np_utils.to_categorical(ids, len(uniques))
        else:
            return encoded_y

