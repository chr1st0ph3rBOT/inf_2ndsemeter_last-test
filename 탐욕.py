llist = [500, 100, 50, 10]

n = 3000

ans = 0

for i in llist:
    if n%i == 0:
        ans = n/i
        break
    ans == n//i
    n = n%i

print(ans)
