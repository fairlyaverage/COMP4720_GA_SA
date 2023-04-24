import time, math
# def progress():
#     LINE_UP, LINE_CLEAR = '\033[1A', '\x1b[2K'
#     TAIL, HEAD, progress = '-', '>', ''
#     i = 0
#     while True:
#         progress=''
#         time.sleep(1)
#         i+=1
#         for k in range(i):
#             progress += TAIL
#         progress += HEAD
#         print(LINE_UP, end=LINE_CLEAR)
#         print(f"{progress}")

EMPTY = '░'
FULL = '█'

# ex: [██████████░░░░░░░░░░] 50.0%
#     container = []
# print(f"[{progress}{remainder}] {percent}")
# LINE_UP end with LINE_CLEAR

def progress_info():
    global progress_info_log, progress_info_intervals
    progress_info_log = []
    progress_info_intervals = []

    # [ 123512059, 123521359, 1235213512 ]
    # want [1] - [0] = interval
def update_progress_info(part, whole):

    current_time = time.time()
    average_interval = math.inf
    time_to_completion = math.inf
    estimated_time_to_completion = math.inf
    elapsed = 0

    progress_info_log.append(current_time)

    # add interval delta
    if len(progress_info_log) > 1: progress_info_intervals.append(current_time - progress_info_log[-2])
    else: progress_info_intervals.append(0)

    elapsed = sum(progress_info_intervals)
    average_interval = progress_info_intervals[-1] # elapsed / len(progress_info_intervals)
    estimated_time_to_completion = (whole - part) * average_interval
    # progress_info_log.append(current_time - progress_info_log[-1])
    estimated_total_time = elapsed + estimated_time_to_completion
    # print(f"intervals = {progress_info_intervals}\n progress_log={progress_info_log}\n")
    return average_interval, elapsed, estimated_time_to_completion, estimated_total_time

def progress(part, whole):
    whole -= 1 # if exclusive range e.g. i in range(10) only goes from 0-9
    if part == 0: progress_info()
    MAX = 50 # + '[] 100.0%' 8-9 extra chars
    # special characters
    EMPTY = '░'
    FULL = '█'
    LINE_UP, LINE_CLEAR = '\033[1A', '\x1b[2K'

    percent = round((part / whole) * 100, 1) # float .1
    average_interval, elapsed, estimated_time_to_completion, estimated_total_time = update_progress_info(part, whole)

    # progress indicator
    # ██████████ adjusted to 50 chars
    complete = FULL * math.floor(percent * .5) # *.5 because fixing for 50 characters x/50 = 50/100
    # ░░░░░░░░░░ adjusted to 50 chars
    remaining = EMPTY * math.ceil((100 - percent) * .5) # see above

    output = f'[{complete}{remaining}] {percent}% | avg: {(average_interval*1000):.0f}ms elapsed: {(elapsed):.0f}s remaining: {(estimated_time_to_completion):.3f}s / {(estimated_time_to_completion / 60):.0f}m / {(estimated_time_to_completion / (60 * 60)):.0f}h total: {(estimated_total_time):.3f}s'
    print(LINE_CLEAR, end=LINE_UP)
    print(output)
    if part == whole: print(LINE_CLEAR, end=LINE_UP)

# delta = 60 # sec, for debug
# for i in range(delta):
#     time.sleep(0.725)
#     progress(i, delta)
