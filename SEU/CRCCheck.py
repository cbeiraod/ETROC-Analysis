from crc import Calculator, Crc8, Configuration

config = Configuration(
    width=8,
    polynomial=0x2F, # Normal representation
    #polynomial=0x97, # Reversed reciprocal representation
    init_value=0x00,
    final_xor_value=0x00,
    reverse_input=False,
    reverse_output=False,
)

CRCcalculator = Calculator(config, optimized=True)

## Real data, 5 full consecutive data frames from an example run
data_list = [
    {
        'data': [
        0b00111100, 0b01011100, 0b00100101, 0b11000010, 0b11001111, # Header
        0b10010000, 0b00000111, 0b01010000, 0b11010100, 0b10100010, # Data
        0b10010000, 0b01000111, 0b01111000, 0b11001000, 0b10100001, # Data
        0b10000100, 0b00000111, 0b10100000, 0b11010100, 0b10101010, # Data
        0b10000100, 0b01000111, 0b10011000, 0b11011000, 0b10101010, # Data
        0b01011111, 0b11000011, 0b11000000, 0b00000100,             # Trailer without CRC
        ],
        'CRC': 0b11101111,
    },
    {
        'data': [
        0b00111100, 0b01011100, 0b00100101, 0b11000010, 0b11001111, # Header
        0b10010000, 0b00000111, 0b01100000, 0b11011000, 0b10100001, # Data
        0b10010000, 0b01000111, 0b01010000, 0b11010000, 0b10100001, # Data
        0b10000100, 0b00000111, 0b10000000, 0b11100000, 0b10101001, # Data
        0b10000100, 0b01000111, 0b10000000, 0b11100000, 0b10101000, # Data
        0b01011111, 0b11000011, 0b11000000, 0b00000100,             # Trailer without CRC
        ],
        'CRC': 0b10111111,
    },
    {
        'data': [
        0b00111100, 0b01011100, 0b00100110, 0b00000010, 0b11001111, # Header
        0b10010000, 0b00000111, 0b01011000, 0b11010100, 0b10100001, # Data
        0b10010000, 0b01000111, 0b01111000, 0b11001000, 0b10100001, # Data
        0b10000100, 0b00000111, 0b10100000, 0b11011000, 0b10101010, # Data
        0b10000100, 0b01000111, 0b10011000, 0b11011000, 0b10101010, # Data
        0b01011111, 0b11000011, 0b11000000, 0b00000100,             # Trailer without CRC
        ],
        'CRC': 0b10010101,
    },
    {
        'data': [
        0b00111100, 0b01011100, 0b00100110, 0b00000010, 0b11001111, # Header
        0b10010000, 0b00000111, 0b01100000, 0b11011000, 0b10100001, # Data
        0b10010000, 0b01000111, 0b01010000, 0b11001100, 0b10100001, # Data
        0b10000100, 0b00000111, 0b10000000, 0b11011100, 0b10101000, # Data
        0b10000100, 0b01000111, 0b10000000, 0b11100000, 0b10101000, # Data
        0b01011111, 0b11000011, 0b11000000, 0b00000100,             # Trailer without CRC
        ],
        'CRC': 0b10101000,
    },
    {
        'data': [
        0b00111100, 0b01011100, 0b00100110, 0b01000010, 0b11001111, # Header
        0b10010000, 0b00000111, 0b01011000, 0b11010100, 0b10100001, # Data
        0b10010000, 0b01000111, 0b01111000, 0b11001000, 0b10100001, # Data
        0b10000100, 0b00000111, 0b10100000, 0b11011000, 0b10101011, # Data
        0b10000100, 0b01000111, 0b10010000, 0b11011100, 0b10101011, # Data
        0b01011111, 0b11000011, 0b11000000, 0b00000100,             # Trailer without CRC
        ],
        'CRC': 0b00110010,
    },
]

for entry in data_list:
    raw_data = entry['data']
    crc = entry['CRC']
    data = bytes(raw_data)
    #data = bytes(raw_data + [crc])
    check = CRCcalculator.checksum(data)


    print("Raw data:")
    print_string = ""
    for dat in raw_data:
        print_string += f"{dat:08b} "
    print(print_string)

    print(f"CRC: {crc:08b}")
    print(f"CRC Check: {check:08b}")