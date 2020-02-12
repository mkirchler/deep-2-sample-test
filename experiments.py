from deeptest import *
from load_tests import *

RESULTS_DIR = 'results/'

def main():
    # run e.g. DFDA on all data sets and save results to {RESULTS_DIR}/dfda_... via
    dev = 'cuda:0'
    run_all_deep(method='dfda', n_runs=100, dev=dev, s_temp='dfda_%s')
    return

def run_all_deep(method='dmmd_kernel', s_temp='dmmd_kernel_med_%s', dev='cpu', n_runs=1000):
    '''Perform all experiments for one deep-learning based test

    # Parameters
    method (str): which test procedure to perform, should be in {'dfda', 'dmmd', 'dmmd_kernel', 'c2st'}
    s_temp (str): template string where within RESULTS_DIR to save each result
    dev (str): which cpu/cuda device to use
    n_runs (int): number of runs to evaluate the test
    '''
    T = get_resnet_test(method, device=dev)

    print('starting birds')
    m_birds = [10, 15, 20, 25, 50, 60]
    res_birds = run_birds_experiment(T, method='deep', n_runs=n_runs, alpha=0.05, ms=m_birds)
    s = s_temp % 'birds.pt'
    s = os.path.join(RESULTS_DIR, s)
    torch.save(res_birds, s)

    print('starting planes')
    m_planes = [10, 15, 20, 25, 50, 75, 100, 150, 200]
    res_planes = run_aircraft_experiment(T, method='deep', n_runs=n_runs, alpha=0.05, ms=m_planes)
    s = s_temp % 'planes.pt'
    s = os.path.join(RESULTS_DIR, s)
    torch.save(res_planes, s)

    print('starting faces')
    m_faces = [10, 15, 20, 25, 50, 75, 100, 150, 200]
    res_faces = run_faces_experiment(T, method='deep', n_runs=n_runs, alpha=0.05, ms=m_faces)
    s = s_temp % 'faces.pt'
    s = os.path.join(RESULTS_DIR, s)
    torch.save(res_faces, s)

    print('starting audio')
    T = get_M5_test(method, device=dev)
    m_audio = [10, 15, 20, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
    res_audio = run_audio_experiment(T, n_runs=n_runs, alpha=0.05, ms=m_audio)
    s = s_temp % 'audio.pt'
    s = os.path.join(RESULTS_DIR, s)
    torch.save(res_audio, s)

    print('starting dogs unsupervised')
    T = get_dogstest_unsupervised(method, device=dev)
    m_dogs = [10, 15, 20, 25, 50, 75, 100, 150, 200]
    res_dogs = run_dogs_experiment(T, method='deep', n_runs=n_runs, alpha=0.05, ms=m_dogs)
    s = s_temp % 'dogs_unsup.pt'
    s = os.path.join(RESULTS_DIR, s)
    torch.save(res_dogs, s)

    print('starting dogs supervised')
    T = get_dogstest_supervised(method, device=dev)
    m_dogs = [10, 15, 20, 25, 50, 75, 100, 150, 200]
    res_dogs = run_dogs_experiment(T, method='deep', n_runs=n_runs, alpha=0.05, ms=m_dogs)
    s = s_temp % 'dogs_sup.pt'
    s = os.path.join(RESULTS_DIR, s)
    torch.save(res_dogs, s)

def run_birds_experiment(T, method='deep', ms=[10], n_runs=1000, alpha=0.05):
    '''Evaluate test T on CUB birds data set 

    # Parameters
    T (TwoSampleTest): test to be evaluated
    method (str): "deep" or "kernel", used to specify data preprocessing
    ms (list of int): complete set of sample sizes to evaluate
    n_runs (int): number of runs to evaluate the test
    alpha (float): significance level
    '''
    c0, c1 = '161.Blue_winged_Warbler', '167.Hooded_Warbler'
    if method == 'deep':
        data = ImageData(path_to_data=PATH_TO_BIRDS, c0=c0, c1=c1, target_shape=(224, 224))
    elif method == 'kernel':
        data = ImageData(path_to_data=PATH_TO_BIRDS, c0=c0, c1=c1, target_shape=(48, 48), gray=True)
    else:
        raise NotImplementedError()
    
    return benchmark_pipe(data, T, data_params=('m', ms), n_runs=n_runs, alpha=0.05, verbose=True)

def run_dogs_experiment(T, method='deep', ms=[10], n_runs=1000, alpha=0.05):
    '''Evaluate test T on Stanford Dogs data set 

    # Parameters
    T (TwoSampleTest): test to be evaluated
    method (str): "deep" or "kernel", used to specify data preprocessing
    ms (list of int): complete set of sample sizes to evaluate
    n_runs (int): number of runs to evaluate the test
    alpha (float): significance level
    '''
    c0, c1 = 'n02090721-Irish_wolfhound', 'n02092002-Scottish_deerhound'
    if method == 'deep':
        data = ImageData(path_to_data=PATH_TO_DOGS, c0=c0, c1=c1, target_shape=(224, 224))
    elif method == 'kernel':
        data = ImageData(path_to_data=PATH_TO_DOGS, c0=c0, c1=c1, target_shape=(48, 48), gray=True)
    else:
        raise NotImplementedError()

    return benchmark_pipe(data, T, data_params=('m', ms), n_runs=n_runs, alpha=0.05, verbose=True)

def run_aircraft_experiment(T, method='deep', ms=[10], n_runs=1000, alpha=0.05):
    '''Evaluate test T on Aircraft data set

    # Parameters
    T (TwoSampleTest): test to be evaluated
    method (str): "deep" or "kernel", used to specify data preprocessing
    ms (list of int): complete set of sample sizes to evaluate
    n_runs (int): number of runs to evaluate the test
    alpha (float): significance level
    '''
    c0, c1 = 'Boeing-737', 'Boeing-747'
    if method == 'deep':
        data = ImageData(path_to_data=PATH_TO_PLANES, c0=c0, c1=c1, target_shape=(224, 224))
    elif method == 'kernel':
        data = ImageData(path_to_data=PATH_TO_PLANES, c0=c0, c1=c1, target_shape=(48, 48), gray=True)
    else:
        raise NotImplementedError()

    return benchmark_pipe(data, T, data_params=('m', ms), n_runs=n_runs, alpha=0.05, verbose=True)

def run_faces_experiment(T, method='deep', ms=[10], n_runs=1000, alpha=0.05):
    '''Evaluate test T on KDEF data set

    # Parameters
    T (TwoSampleTest): test to be evaluated
    method (str): "deep" or "kernel", used to specify data preprocessing
    ms (list of int): complete set of sample sizes to evaluate
    n_runs (int): number of runs to evaluate the test
    alpha (float): significance level
    '''
    if method == 'deep':
        data = get_faces_data(gray=False, target_shape=(224, 224), cropping=False)
    elif method == 'kernel':
        data = get_faces_data(gray=False, target_shape=(96, 96), cropping=True)
    else:
        raise NotImplementedError()

    return benchmark_pipe(data, T, data_params=('m', ms), n_runs=n_runs, alpha=0.05, verbose=True)

def run_audio_experiment(T, ms=[10], n_runs=1000, alpha=0.05):
    '''Evaluate test T on AM audio data set

    # Parameters
    T (TwoSampleTest): test to be evaluated
    ms (list of int): complete set of sample sizes to evaluate
    n_runs (int): number of runs to evaluate the test
    alpha (float): significance level
    '''
    c0, c1 = '04 Gramatik - Pardon My French.wav', '05 Gramatik - We Used To Dream.wav'
    data = AMData(c0, c1, PATH_TO_AUDIO, noise_level=1., d=1000)

    return benchmark_pipe(data, T, data_params=('m', ms), n_runs=n_runs, alpha=0.05, verbose=True)

if __name__ == '__main__':
    main()
