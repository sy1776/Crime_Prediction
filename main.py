from import_data import run_import
from manipulate_data import run_manipulate_data
from ml_models import run_ml_models
import time

DISPLAY = True

def main():
    start_time = time.time()
    if DISPLAY:
        print("Start = %s" % (time.ctime()) )
        print("______________________________________________")
        print("")

    run_import()
    run_manipulate_data()
    run_ml_models()

    duration = time.time() - start_time
    if DISPLAY:
        print("______________________________________________")
        print("")
        print("End= %s, Duration= %f" % (time.ctime(), duration))

if __name__ == "__main__":
    main()