## Import
from glob import glob
import os

def find_pattern_multiple_indexs(bitstream):
    pattern = "3c5c"
    pattern_length = len(pattern) * 4  # 4 bits per hexadecimal digit

    indices = []  # List to store the indices of pattern occurrences

    for i in range(len(bitstream) - pattern_length + 1):
        current_substring = bitstream[i:i+pattern_length]
        hexadecimal = hex(int(current_substring, 2))[2:]  # Convert binary to hexadecimal
        if hexadecimal == pattern:
            indices.append(i)

    return indices  # Return the list of indices

### Translate binary to readable data - mutiple indices
def translate_with_indices(input_stream, positions, parent_dir, output, chipID, suppress=False):
    f = open(parent_dir+'/'+output, 'w')
    residual = ''

    # hex chipID to binary
    binID = format(int(chipID, 0), '017b')
    wordlength = 40
    printline = ''
    key = ''

    for i, index in enumerate(positions):

        word = input_stream[index:index+wordlength]
        printline = "ETROC2 0 "# + "{:d} ".format(channel)

        if len(word) != 40:
            residual = word
            break
        elif len(word) == 40:
            # Header
            if word[0:16] == '0011110001011100' and word[16:18] == '00':
                printline += "HEADER "
                printline += "L1COUNTER " + word[18:26] + " "
                printline += "TYPE " + word[26:28] + " "
                printline += "BCID " + f"{int(word[28:40], base=2)}" + "\n"
                key = 'header'
            # Frame filler
            elif word[0:16] == '0011110001011100' and word[16:18] == '10':
                printline += "FRAMEFILLER "
                printline += "L1COUNTER " + word[18:26] + " "
                printline += "EBS " + word[26:28] + " "
                printline += "BCID " + f"{int(word[28:40], base=2)}" + "\n"
                key = 'filler'
            # Firmware filler
            elif word[0:16] == '0011110001011100' and word[16:18] == '11':
                printline += "FIRMWAREFILLER "
                printline += "MISSINGCOUNT " + word[18:40] + "\n"
                key = 'filler'
            else:
                printline += "NOT DEFINED " + word[0:16] + " " + word[16:18] + " " + word[18:] + "\n"
                key = 'whatisthis'
                pass

        # Save if the data is filler
        if key == 'filler':
            f.write(printline)

        # if the data is header, try to find the data and trailer 
        elif key == 'header':
            try:
                loop = int((positions[i+1] - positions[i])/40)
                for k in range(1, loop):
                    word = input_stream[index+(wordlength*k):index+(wordlength*(k+1))]
                    # Trailer
                    if word[0:18] == '0'+str(binID):
                        printline += "ETROC2 0 TRAILER "
                        printline += "CHIPID " + f"{hex(int(word[1:18], base=2))}" + " "
                        printline += "STATUS " + word[18:24] + " "
                        printline += "HITS " + f"{int(word[24:32], base=2)}" + " "
                        printline += "CRC " + word[32:40] + "\n"
                        key = 'trailer'

                    elif word[0] == '1':
                        printline += "ETROC2 0 DATA "
                        printline += "EA " + word[1:3] + " "
                        printline += "COL " + "{:d} ".format(int(word[3:7], base=2))
                        printline += "ROW " + "{:d} ".format(int(word[7:11], base=2))
                        printline += "TOA " + "{:d} ".format(int(word[11:21], base=2))
                        printline += "TOT " + "{:d} ".format(int(word[21:30], base=2))
                        printline += "CAL " + "{:d} ".format(int(word[30:40], base=2)) + "\n"
                        key = 'data'

                    # if the data is trailer, write the output in the file
                    if key == 'trailer':
                        f.write(printline)
            except:
                # out of index range, move to the next file
                residual = word

    f.close()
    return residual


def main():
    import argparse

    parser = argparse.ArgumentParser(
                    prog='PlaceHolder',
                    description='offline translate script',
                    #epilog='Text at the bottom of help'
                    )

    parser.add_argument(
        '-d',
        '--inputdir',
        metavar = 'DIRNAME',
        type = str,
        help = 'input directory name',
        required = True,
        dest = 'dirname',
    )

    args = parser.parse_args()

    ### Let's convert!
    # list directories that want to scan
    dirs=str(args.dirname)
    outdir = dirs+'/output'
    os.mkdir(outdir)

    residual = ''
    files = glob(dirs+'/*Data_[0-9]*.dat')
    files = sorted(files)

    for i, ifile in enumerate(files):
        residual = ''
        # if i > 2: break
        # Let's make a very long bitstream single line
        bitstream = residual + ''
        with open(ifile, 'r') as infile:
            # make data bitstream in a single line
            for line in infile.readlines():
                if line[0:4] == '1100':
                    bitstream += line.strip()[4:]

        positions = find_pattern_multiple_indexs(bitstream)
        outname = 'TDC_Data_translated_'+str(i)+'.dat'
        residual = translate_with_indices(bitstream, positions, outdir, outname, '0x17f0f')

if __name__ == "__main__":
    main()
