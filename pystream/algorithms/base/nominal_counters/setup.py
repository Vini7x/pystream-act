def configuration(parent_package='',top_path=None):
    import os
    from os.path import join

    import numpy
    from numpy.distutils.misc_util import Configuration


    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('nominal_counters', parent_package, top_path)

    dirs = [numpy.get_include(), '.']

    config.add_extension(name='nominal_counter',
                         sources=['nominal_counter.pyx'],
                         libraries=libs,
                         include_dirs=dirs)

    config.make_config_py() # installs __config__.py
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
