llist = [500, 100, 50, 10]

n = 3000

ans = 2^1010101010101

def sol(s, k): # s는 누적합, k는 사용한 동전 개수
    global ans
    if k > ans:
        return
    if s == n:
        if ans > k:
            ans = k
        return
    if s > n:
        return

    for i in llist:
        sol(s+i, k+1)
    return

for i in llist:
    sol(i, 1)
print(ans)
