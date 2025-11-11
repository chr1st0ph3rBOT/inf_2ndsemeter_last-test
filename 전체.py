llist = [10, 50, 100, 500]

n = 1000

ans = 2^1010101010101

def sol(s, k): # s는 누적합, k는 사용한 동전 개수
    global ans
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
