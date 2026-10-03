import json, math
d = json.load(open(r'PROJECT_ROOT\data\l3_mappo_typeaware_result.json', encoding='utf-8'))
per = d['per_seed']
gaps = [v['gap']*100 for v in per.values()]
n = len(gaps)
mean = sum(gaps)/n
sd = math.sqrt(sum((v-mean)**2 for v in gaps)/(n-1))

def welch(m1, s1, n1, m2, s2, n2):
    t = (m1-m2)/math.sqrt(s1*s1/n1+s2*s2/n2)
    df = (s1*s1/n1+s2*s2/n2)**2 / ((s1*s1/n1)**2/(n1-1)+(s2*s2/n2)**2/(n2-1))
    return t, df

t1, df1 = welch(mean, sd, n, 10.23, 0.32, 10)
t2, df2 = welch(mean, sd, n, 5.53, 1.54, 10)
with open(r'PROJECT_ROOT\data\_welch_typeaware.txt', 'w') as f:
    f.write('exact mean=%.4f std=%.4f n=%d\n' % (mean, sd, n))
    f.write('vs anonymous MAPPO 10.23+/-0.32 (n=10): t=%.2f df=%.1f\n' % (t1, df1))
    f.write('vs lambda* type-aware 5.53+/-1.54 (n=10): t=%.2f df=%.1f\n' % (t2, df2))
print('written')
