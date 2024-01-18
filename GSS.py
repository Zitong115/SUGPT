def mix(a, b, c):
    a -= b
    a -= c
    a ^= (c >> 13)
    b -= c
    b -= a
    b ^= (a << 8)
    c -= a
    c -= b
    c ^= (b >> 13)
    a -= b
    a -= c
    a ^= (c >> 12)
    b -= c
    b -= a
    b ^= (a << 16)
    c -= a
    c -= b
    c ^= (b >> 5)
    a -= b
    a -= c
    a ^= (c >> 3)
    b -= c
    b -= a
    b ^= (a << 10)
    c -= a
    c -= b
    c ^= (b >> 15)
    return a, b, c

def BOB1(str, length):
    a = b = 0x9e3779b9
    c = 2

    while length >= 12:
        a += (ord(str[0]) + (ord(str[1]) << 8) + (ord(str[2]) << 16) + (ord(str[3]) << 24))
        b += (ord(str[4]) + (ord(str[5]) << 8) + (ord(str[6]) << 16) + (ord(str[7]) << 24))
        c += (ord(str[8]) + (ord(str[9]) << 8) + (ord(str[10]) << 16) + (ord(str[11]) << 24))
        a, b, c = mix(a, b, c)
        str = str[12:]
        length -= 12

    c += length
    if length == 11:
        c += (ord(str[10]) << 24)
    if length >= 10:
        c += (ord(str[9]) << 16)
    if length >= 9:
        c += (ord(str[8]) << 8)
    if length >= 8:
        b += (ord(str[7]) << 24)
    if length >= 7:
        b += (ord(str[6]) << 16)
    if length >= 6:
        b += (ord(str[5]) << 8)
    if length >= 5:
        b += ord(str[4])
    if length >= 4:
        a += (ord(str[3]) << 24)
    if length >= 3:
        a += (ord(str[2]) << 16)
    if length >= 2:
        a += (ord(str[1]) << 8)
    if length >= 1:
        a += ord(str[0])
    a, b, c = mix(a, b, c)
    return c
