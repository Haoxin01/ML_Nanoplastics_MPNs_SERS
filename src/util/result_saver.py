import os

def build_result_dir(sys_path):
    # build result directory if not exist
    result_dir = sys_path + '\\result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # build visualization result directory if not exist
    visualization_dir = result_dir + '\\visualization'
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # build model result directory if not exist
    model_dir = result_dir + '\\model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

