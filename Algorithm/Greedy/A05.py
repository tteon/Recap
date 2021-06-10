# n ; 공의 갯수
# m ; 각 공의 무게

# 무게마다 볼링공이 몇 개 있는지를 계산해야함.
n, m = map(int, input().split())
data = list(map(int, input().split()))

#
array = [0] * 11 # 무게가 1~10까지만 존재 가능함.
for x in data:
    array[x] += 1 # 무게에 해당하는 볼링공 개수 count

result = 0

# A가 특정한 무게 볼링공을 선택했을 때, 이어서 B가 볼링공을 선택하는 경우를 차례대로 계산.
for i in range(1, m+1):
    n -= array[i] # 무게가 i인 볼링공의 개수(A가 선택할 수 있는 개수) 제외
    result += array[i] * n # B가 선택하는 경우의 수와 곱하기.

print(result)