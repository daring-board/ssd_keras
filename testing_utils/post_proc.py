
f = open('./ssd_result.csv', 'r')
lines = [l for l in f]
flags_dict = {l.split(',')[0]: l.split(',')[1].split(':')  for l in lines[1:]}

result_dict = {}
for key in flags_dict.keys():
    flags = flags_dict[key]
    tmp, max_len = 0, 0
    for flg in flags:
        flg = int(flg)
        if flg == 0:
            if max_len < tmp:
                max_len = tmp
            tmp = 0
        else:
            tmp += flg
    print(max_len)
    result_dict[key] = 1 if max_len >= 6 else 0

with open('./result.csv', 'w') as f:
    for key in result_dict.keys():
        f.write('%s,%d'%(key, result_dict[key]))
