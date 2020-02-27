import random
import math


def ap_k(relev_positions, k):
    ans = 0
    num_rel = 0
    for rl in relev_positions:
        if rl < k:
            num_rel += 1
            ans += num_rel / rl
        else:
            break
    return ans


def ndcg(relev_positions, k):
    ans = 0
    for rl in relev_positions:
        if rl < k:
            ans += 1 / math.log(rl + 1, 2)
        else:
            break
    return ans


a = [[random.randint(1, 20)] for i in range(300)]
a.extend([[random.randint(1, 3)] for i in range(1000)])
a.extend([[random.randint(1, 100)] for i in range(30)])
sm= 0
for k in a:
    for i in k:
        sm += i
print(sm/len(a)) #5.27
apk, nd = 0, 0
for s in a:
    apk += ap_k(s, 10)
    nd += ndcg(s, 10)
apk /= len(a)
nd /= len(a)
print("apk: ", apk) #0.49
print("ndcg: ", nd) #0.58
