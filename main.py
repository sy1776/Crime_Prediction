from import_clean_data import run_import_clean
from manipulate_data import run_manipulate_data
from exp_smoothing_model import run_es_model
from ml_models import run_ml_models
from predict import run_preidct
import time

DISPLAY = True

def main():
    start_time = time.time()
    if DISPLAY:
        print("Start = %s" % (time.ctime()) )
        print("______________________________________________")
        print("")

    run_import_clean()
    #run_manipulate_data()
    #run_es_model()
    #run_ml_models()
    #run_preidct()

    duration = time.time() - start_time
    if DISPLAY:
        print("______________________________________________")
        print("")
        print("End= %s, Duration= %f" % (time.ctime(), duration))

if __name__ == "__main__":
    main()