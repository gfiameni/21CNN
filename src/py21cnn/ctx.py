def init():
    global inputs, HP, Data, model
    global filepath, logdir, main_process
    global fit_options, compile_options, generators
    global test_data
    
    inputs = None # argparse.Namespace object, with all input variables
    HP = None # py21cnn.utilities.AuxiliaryHyperparameters object
    Data = None # py21cnn.utilities.Data or py21cnn.utilities.LargeData object
    model = None # keras model

    filepath = None # data_location + filename for model, summary, logs
    logdir = None # location of Tensorboard logs
    main_process = None # bool, main process saves models, summary, logs to disk, writes to stdout
    
    fit_options = None # dictionary passed to model.fit, model.fit_generator
    compile_options = None # dictionary passed to model.compile
    generators = None # in the case of large_run, dictionary of train, validation, test generators

    test_data = None # in the case of large_run, saving the actual filenames and "labels" is needed for prediction