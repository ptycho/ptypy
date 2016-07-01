'''
some utilities to help with writing tests
'''


    path = inspect.stack()[0][1]
    return '/'.join(os.path.split(path)[0].split(os.sep)[:-2] +
                    ['test_data/process_lists', name])