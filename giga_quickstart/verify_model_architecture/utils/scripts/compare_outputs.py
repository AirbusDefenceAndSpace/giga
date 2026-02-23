def compare_outputs(py_file, cpp_file):
    """
    Compares two output files (Python and C++) line by line, of the giga and the python inference
    Returns True if all differences are within 0.0001, False otherwise.
    """

    no_errors = True    
    with open(py_file) as f1, open(cpp_file) as f2:
        d1 = [float(x) for x in f1.read().split()]
        d2 = [float(x) for x in f2.read().split()]

    #print(f"{'Index':<8} | {'Python':<10} | {'C++':<10} | {'Diff'}")
    for i, (a, b) in enumerate(zip(d1, d2)):
        diff = abs(a - b)
        if diff >= 0.0001:
            #print(f"{i:<8} | {a:<10.4f} | {b:<10.4f} | {diff:.4f}")
            no_errors = False
    
    if no_errors:
        print("✅✅✅✅✅✅Converted successfully. No errors found that were bigger then 0.0001✅✅✅✅✅")
        return True
    else: 
        print("❌❌❌❌❌ Errors bigger then 0.0001 in the code ❌❌❌❌❌")
        return False