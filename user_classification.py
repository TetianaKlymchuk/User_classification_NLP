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

