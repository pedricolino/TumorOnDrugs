"""Runtime environment helpers."""


def check_prepare_gpu(verbose: bool = True) -> bool:
    import shutil
    import scanpy as sc

    try:
        import cupy  # noqa: F401
        have_cupy = True
    except Exception:
        have_cupy = False
    try:
        import rapids_singlecell as rsc
        have_rsc = True
    except Exception:
        have_rsc = False

    use_gpu = have_cupy and have_rsc and shutil.which("nvidia-smi") is not None
    if verbose:
        if use_gpu:
            print(f"Running on GPU using RAPIDS Single Cell version {rsc.__version__}")
        else:
            print("Running on CPU using Scanpy CPU version.", sc.__version__)
    return use_gpu


def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
    except NameError:
        print("Could not detect IPython environment. Assuming standard Python interpreter.")
        return False

    try:
        import __main__ as main_module
        if hasattr(main_module, '__marimo__'):
            print("Running in Marimo environment.")
            return True
    except ImportError:
        main_module = None

    if shell == "ZMQInteractiveShell":
        print("Running in Jupyter notebook.")
        return True
    if shell == "TerminalInteractiveShell":
        print("Running in terminal.")
        return False
    print("Running in an unknown environment.")
    return False
