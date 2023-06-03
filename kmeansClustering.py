from mpl_toolkits.mplot3d import Axes3D
import sys
import matplotlib.pyplot as plt
import numpy as np

# x : clustering 해야할 vectors / type : list of ndarray / len(x) = n 
# z : group representatives / type : list of ndarray / len(z) = k
# c : 각 vector가 속한 group 번호 / type : list of int / len(c) = n

# 해당 상태에서의 Jclust 값을 반환한다.
cal_Jclust = lambda x, z, c: sum(np.linalg.norm(x[idx] - z[c[idx]])**2 for idx in range(len(x))) / len(x)

# 각 vector(x)를 주어진 representative(z)를 이용해 Jclust에서의 term값이 최소화 되도록 각 group(c)에 할당한다.
def partition(x, z, c):
    for idx in range(0, len(x)):
        assign_group = 0  # 일단 먼저 group 0 에 할당 하자.
        best = np.linalg.norm(x[idx]-z[0])**2
        for group in range(1, k): # group 1 ~ k-1까지 중에 더 나은 선택을 한다.
            cal_term = np.linalg.norm(x[idx]-z[group])**2
            if cal_term < best:
                best = cal_term
                assign_group = group
        c[idx] = assign_group

# 각 vector(x) 들이 group(c)에 할당 되었을 때 각각의 group에 속한 vector들로
# representative(z)를 update한다. (centroid 이용)
def update_representative(x, z, c):
    group_sum = [np.zeros(len(x[0])) for idx in range(0, k)]  # 각 group에 속한 vector들을 더한값(element-wise) 
    group_size = np.zeros(k, dtype = int) # 각 group에 속한 vector들의 수

    # 각 vector는 자기가 속한 group에 해당되는 group_sum을 기여하고 해당 group_size를 1 증가시킨다.
    for idx in range(0, len(x)):
        group_sum[c[idx]] += x[idx]
        group_size[c[idx]]+=1
    
    # 각 group_sum과 각 group_size를 구했으면 이를 이용해 centroid를 찾고 update 한다.
    for group in range(0, k):
        if group_size[group]!=0:
            z[group] = group_sum[group]/group_size[group]  # centroid

#------------main------------#

# $ python .\kmeansClustering.py .\inputfile.txt [k] [max_iter] 
input_source = sys.argv[1]
k = int(sys.argv[2])
max_iter = int(sys.argv[3])

sys.stdin = open(input_source)

x = []
unique_vectors = set() # inputfile.txt로 받는 vector들에서 중복허용 X 
# -> 그냥 x의 원소들 중 임의로 몇 개 뽑아서 initial representative 설정시 
# 같은 representative가 발생할 수 있고, 이 경우 어떤 cluster에 속한 vector가 0이 되는 경우가 발생가능. 이를 방지하기 위함

# inputfile.txt 로부터 vector x 초기화
while True:
    vec = sys.stdin.readline()
    if vec == "":
        break
    x.append(np.array(list((map(float, vec.split())))))
    unique_vectors.add(tuple(list((map(float, vec.split())))))

# initial representative 설정
z = []

# 서로 다른 값을 가지는 vector들(즉, 위 코드에서 unique_vectors) 중에서 k개를 골라서 representative로 설정
# len(unique_vectors) >= k 이다. (서로 다른 벡터 n개를 n+1개 이상의 cluster로 분할하는 입력은 유효하지 않기 때문)
for e in unique_vectors:
    if len(z)==k: # k개의 group에 대해서 initial representative 설정시 break
        break
    z.append(np.array(list(e)))

# 각 vector가 속한 group 번호를 나타내는 c 초기화 (일단 먼저 0으로 초기화, partition에서 값 결정)
c = [0 for i in range(len(x))]

prev_Jclust_value = -1 # 이전 iteration에서의 Jclust값
actual_iter = max_iter # 실제 iteration한 횟수 (최대 max_iter)

for i in range(1, max_iter+1):
    partition(x, z, c)
    update_representative(x, z, c)

    now_Jclust_value = cal_Jclust(x, z, c) 

    if prev_Jclust_value == now_Jclust_value: # 이전 iteration에서의 Jclust와 같을 경우(즉, convergence) break
        actual_iter = i
        break
    else:
        prev_Jclust_value = now_Jclust_value

# 결과 출력
print("# of actual iteration :", actual_iter)

print("representative : ", end = "")
for group_representative in z:
    print("(", end="")
    for idx in range(0, len(group_representative)):
        if idx == len(group_representative)-1:
            print(group_representative[idx], end="")
        else:
            print(group_representative[idx], end=", ")
    print(")", end=" ")
print()

group_size = np.zeros(k, dtype=int) # 각 group(cluster)에 속한 vector의 수
for e in c:
    group_size[e] += 1
    
for group in range(k):
    print("# of vectors for cluster", group+1, ":", group_size[group])


n = 100
xmin, xmax, ymin, ymax, zmin, zmax = 0, 20, 0, 20, 0, 20
cmin, cmax = 0, 2

xs = np.zeros(100)
ys = np.zeros(100)
zs = np.zeros(100)
color = np.zeros(100)

for idx in range(len(x)):
    xs[idx] = x[idx][0]
    ys[idx] = x[idx][1]
    zs[idx] = x[idx][2]
    color[idx] = c[idx]+1

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
#ax.patch.set_facecolor('#a5e6b6')

for idx in range(0, k):
    mark = "$" + str(idx+1) + "$"
    ax.scatter(z[idx][0], z[idx][1], z[idx][2], c = 'k', marker=mark, s=40)

ax.scatter(xs, ys, zs, c=color, marker='o', s=15)

plt.show()
