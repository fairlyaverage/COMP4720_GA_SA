import time, math
def progress():
    LINE_UP, LINE_CLEAR = '\033[1A', '\x1b[2K'
    TAIL, HEAD, progress = '-', '>', ''
    i = 0
    while True:
        progress=''
        time.sleep(1)
        i+=1
        for k in range(i):
            progress += TAIL
        progress += HEAD
        print(LINE_UP, end=LINE_CLEAR)
        print(f"{progress}")

EMPTY = '░'
FULL = '█'

# ex: [██████████░░░░░░░░░░] 50.0%
#     container = []
# print(f"[{progress}{remainder}] {percent}")
# LINE_UP end with LINE_CLEAR

def progress(part, whole):
    MAX = 50 # + '[] 100.0%' 8-9 extra chars
    # special characters
    EMPTY = '░'
    FULL = '█'
    LINE_UP, LINE_CLEAR = '\033[1A', '\x1b[2K'

    percent = round((part / whole) * 100, 1) # float .1

    # progress indicator
    # ██████████ adjusted to 50 chars
    complete = FULL * math.floor(percent * .5) # *.5 because fixing for 50 characters x/50 = 50/100
    # ░░░░░░░░░░ adjusted to 50 chars
    remaining = EMPTY * math.ceil((100 - percent) * .5) # see above

    output = f'[{complete}{remaining}] {percent}%]'
    print(LINE_CLEAR, end=LINE_UP)
    print(output)
