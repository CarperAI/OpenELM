from walk_creator import walker_creator

def make_CPPN():
    """ Make a CPPN based walker"""
    return

def make_walker():
    wc = walker_creator()

    # the main body is based on a CPPN
    sides = make_CPPN()

    return wc.get_walker()