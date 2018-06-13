
import argparse
import os
import glob

import numpy as np

from scipy.io.wavfile import read

from VAD import Neural_Network_VAD


def _resolveargs():
    parser = argparse.ArgumentParser(
        description='This scripts performs Speech/Non-Speech'
        + 'idenftification(SNI) for .WAV files, saves the plot as png'
        + ' and the SNI results in an csv file')
    parser.add_argument('name', metavar='wav file./folder names',
                        type=str, nargs='+', help='wav file locations '
                        + 'ending in .wav or folder location with a '
                        + 'file seperator')

    parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                        help='disable verbose')
    parser.set_defaults(verbose=True)
    parser.add_argument('--no-showplt', dest='showplt', action='store_false',
                        help='disable showing plot')
    parser.set_defaults(showplt=True)
    parser.add_argument('--no-saveplt', dest='saveplt', action='store_false',
                        help='disable saving plot')
    parser.set_defaults(saveplt=True)
    parser.add_argument('--no-csvfile', dest='csvfile', action='store_false',
                        help='disable saving result in csvfile')
    parser.set_defaults(csvfile=True)
    args = parser.parse_args()
    return args


def _resolvefilenames(files):
    filenames = []
    for name in files:
        if os.path.isdir(name):
            folderfiles = glob.glob(name+'*.wav')
            if len(folderfiles) == 0:
                print('No wav files in location %s' % (name))
            else:
                filenames.extend(folderfiles)
        else:
            if '.wav' in name:
                filenames.append(name)
            else:
                print('The filename %s does not have .wav extension' % (name))
    return filenames


def main():
    args = _resolveargs()
    filenames = _resolvefilenames(args.name)

    for file in filenames:
        fs, speech = read(file)
        if type(speech) == np.int16:
            speech /= 2**(15)
        if type(speech) == np.int32:
            speech /= 2**(31)

        if args.verbose:
            print('processing %s' % (file))
            name = file.split(os.sep)[-1]
        Neural_Network_VAD(speech, fs, name, show_plt=args.showplt,
                           plt_save=args.saveplt, csvfile=args.csvfile)


if __name__ == '__main__':
    main()
