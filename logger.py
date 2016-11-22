import logging
import os

def createLogger(filename, description = "", printToConsole=True):
    log = logging.getLogger(description)
    log.setLevel(logging.DEBUG)
    
    # create formatter
    formatter = logging.Formatter('%(asctime)s :  %(levelname)s : %(message)s')
    
    # create file handler
    path = filename.split("/")
    if path[-1] == '': # filename ends in '/'
        directory = "/".join(path[:-2])
    else:
        directory = "/".join(path[:-1])

    if not os.path.exists(directory):
        print("Directory specified for logs doesn't exist! Creating directory'" + directory + "/'")
        os.makedirs(directory)

    filehandler = logging.FileHandler(filename)
    filehandler.setFormatter(formatter)

    log.addHandler(filehandler)

    # create stream handler
    if printToConsole:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        log.addHandler(ch)
   
    return log
